//! Indexer: walk a path, chunk markdown, write rows to the index.
//!
//! No embeddings here — `embedding` is left null; `markq embed` fills it in
//! later. Indexing is incremental and keyed on each file's `content_hash`:
//! unchanged files are skipped (keeping their existing rows and embeddings),
//! edited files have their old chunks deleted before the new ones are added,
//! new files are added, and files that have vanished from disk are pruned.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use arrow_array::{new_null_array, ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::SchemaRef;
use markq_chunker::{chunk_markdown, ApproxTokenizer, ChunkOptions};
use markq_core::{ChunkColumn, Index, SCHEMA_VERSION};
use tracing::{debug, info, warn};
use walkdir::WalkDir;

const DEFAULT_COLLECTION: &str = "default";

/// Outcome of a `markq index <path>` invocation.
pub struct IndexReport {
    /// Files (re)indexed this run — new files plus changed files.
    pub files: usize,
    /// Chunks written this run (from the `files` above).
    pub chunks: usize,
    /// Files skipped because their content was unchanged since the last index.
    pub skipped: usize,
    /// Previously-indexed files pruned because they no longer exist on disk.
    pub removed: usize,
}

pub async fn run_index<I: Index>(idx: &I, root: &Path) -> Result<IndexReport> {
    let root = root
        .canonicalize()
        .with_context(|| format!("canonicalize {}", root.display()))?;

    let schema = idx.arrow_schema();
    // Snapshot what's already indexed so we can skip files whose content is
    // unchanged rather than re-appending duplicate rows.
    let existing = idx
        .existing_file_hashes(DEFAULT_COLLECTION)
        .await
        .context("read existing file hashes")?;

    let mut total_files = 0usize;
    let mut total_chunks = 0usize;
    let mut skipped = 0usize;
    let mut row_files: Vec<FileRows> = Vec::new();
    // Track every path still present on disk so we can prune the rows of files
    // that have since been deleted.
    let mut seen: HashSet<String> = HashSet::new();

    let entries: Vec<_> = WalkDir::new(&root)
        .follow_links(false)
        .into_iter()
        // Skip hidden directories (`.git`, `.venv`, …) and tooling caches
        // (`node_modules`, `target`). Without this, indexing a repo root
        // pulls in vendored READMEs and the index balloons with content
        // the user never meant to retrieve. `WalkDir::filter_entry` prunes
        // whole subtrees, so a hidden dir doesn't even get descended into.
        .filter_entry(|e| !is_excluded(e))
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("md"))
        .collect();

    for entry in entries {
        let path = entry.path();
        let raw = match read_raw_file(path, &root) {
            Ok(raw) => raw,
            Err(e) => {
                warn!(file = %path.display(), error = %e, "skipping file");
                continue;
            }
        };
        seen.insert(raw.path_str.clone());
        match existing.get(&raw.path_str) {
            // Unchanged file: identical bytes already indexed → skip, keeping
            // its existing rows (and embeddings) untouched.
            Some(hash) if *hash == raw.content_hash => {
                skipped += 1;
                continue;
            }
            // Edited file: drop its old chunks before adding the new ones so
            // the old text (and any chunks the shorter version no longer has)
            // don't linger as orphans.
            Some(_) => {
                idx.delete_by_path(DEFAULT_COLLECTION, &raw.path_str)
                    .await
                    .with_context(|| format!("delete prior chunks for {}", raw.path_str))?;
            }
            // New file: nothing to delete.
            None => {}
        }
        let chunks = chunk_raw(&raw);
        total_files += 1;
        total_chunks += chunks.len();
        row_files.push(FileRows {
            path_str: raw.path_str,
            uri: raw.uri,
            content_hash: raw.content_hash,
            mtime_nanos: raw.mtime_nanos,
            chunks,
        });
    }

    // Prune rows for files that were indexed previously but are gone from disk.
    // Scope the prune to the indexed root: a previously-indexed file outside
    // `root` (e.g. from indexing a sibling directory) is out of this run's
    // scope and must be left alone — only files under `root` that have
    // vanished get pruned.
    let mut removed = 0usize;
    for path in existing.keys() {
        if Path::new(path).starts_with(&root) && !seen.contains(path) {
            idx.delete_by_path(DEFAULT_COLLECTION, path)
                .await
                .with_context(|| format!("prune removed file {path}"))?;
            removed += 1;
        }
    }

    if !row_files.is_empty() {
        let batch = build_record_batch(&schema, &row_files)?;
        debug!(rows = batch.num_rows(), "upserting batch");
        idx.upsert_chunks(vec![batch])
            .await
            .context("upsert chunks into index")?;
    } else {
        info!(path = %root.display(), "nothing to index (no new or changed files)");
    }

    Ok(IndexReport {
        files: total_files,
        chunks: total_chunks,
        skipped,
        removed,
    })
}

struct FileRows {
    path_str: String,
    uri: String,
    content_hash: String,
    mtime_nanos: i64,
    chunks: Vec<ChunkRow>,
}

struct ChunkRow {
    id: String,
    chunk_index: i32,
    text: String,
    token_count: i32,
}

/// A file's bytes read once, with identity (uri / path / content hash / mtime)
/// computed but not yet chunked. Splitting the read from the chunk lets the
/// indexer decide skip/replace/add from `content_hash` before paying to chunk.
struct RawFile {
    uri: String,
    path_str: String,
    content_hash: String,
    mtime_nanos: i64,
    text: String,
}

fn read_raw_file(path: &Path, root: &Path) -> Result<RawFile> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes).into_owned();

    let content_hash = blake3::hash(&bytes).to_hex().to_string();
    let mtime_nanos = file_mtime_nanos(path)?;

    let rel = path.strip_prefix(root).unwrap_or(path);
    let rel_str = rel.to_string_lossy().replace('\\', "/");
    let uri = format!("markq://{DEFAULT_COLLECTION}/{rel_str}");
    let path_str = path.to_string_lossy().into_owned();

    Ok(RawFile {
        uri,
        path_str,
        content_hash,
        mtime_nanos,
        text,
    })
}

fn chunk_raw(raw: &RawFile) -> Vec<ChunkRow> {
    let opts = ChunkOptions::default();
    let chunks = chunk_markdown(&raw.text, &opts, &ApproxTokenizer);

    chunks
        .into_iter()
        .map(|c| {
            // Identity = (uri, content_hash, chunk_index). Two distinct files
            // with identical content (duplicated READMEs in a monorepo,
            // vendored notes, generated docs) must not collide on id; the
            // merge-on-id upsert path relies on this to avoid one file
            // overwriting another's chunk text.
            let mut h = blake3::Hasher::new();
            h.update(raw.uri.as_bytes());
            h.update(&[0u8]); // separator so prefix collisions are impossible
            h.update(raw.content_hash.as_bytes());
            h.update(&(c.index as u32).to_le_bytes());
            ChunkRow {
                id: h.finalize().to_hex().to_string(),
                chunk_index: c.index as i32,
                text: c.text,
                token_count: c.token_count.min(i32::MAX as usize) as i32,
            }
        })
        .collect()
}

/// True for entries that should be pruned from the walk. Always permit the
/// root itself (`depth == 0`) so `markq index .markq-test/` still works.
fn is_excluded(e: &walkdir::DirEntry) -> bool {
    if e.depth() == 0 {
        return false;
    }
    let name = match e.file_name().to_str() {
        Some(n) => n,
        None => return false,
    };
    // Hidden (dotfiles / dotdirs). `.` and `..` never appear here — WalkDir
    // doesn't emit them.
    if name.starts_with('.') {
        return true;
    }
    // Common tooling caches that aren't dot-prefixed.
    matches!(name, "node_modules" | "target")
}

fn file_mtime_nanos(path: &Path) -> Result<i64> {
    let meta = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    let mtime = meta
        .modified()
        .with_context(|| format!("mtime {}", path.display()))?;
    let dur = mtime
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| SystemTime::now().duration_since(UNIX_EPOCH).unwrap());
    Ok(dur.as_nanos() as i64)
}

fn build_record_batch(schema: &SchemaRef, files: &[FileRows]) -> Result<RecordBatch> {
    let n: usize = files.iter().map(|f| f.chunks.len()).sum();

    let mut ids = Vec::with_capacity(n);
    let mut collections = Vec::with_capacity(n);
    let mut uris = Vec::with_capacity(n);
    let mut paths = Vec::with_capacity(n);
    let mut hashes = Vec::with_capacity(n);
    let mut mtimes = Vec::with_capacity(n);
    let mut chunk_idx = Vec::with_capacity(n);
    let mut texts = Vec::with_capacity(n);
    let mut tokens: Vec<Option<i32>> = Vec::with_capacity(n);
    let mut context_ids: Vec<Option<String>> = Vec::with_capacity(n);
    let mut schema_versions = Vec::with_capacity(n);

    for f in files {
        for c in &f.chunks {
            ids.push(c.id.clone());
            collections.push(DEFAULT_COLLECTION.to_string());
            uris.push(f.uri.clone());
            paths.push(f.path_str.clone());
            hashes.push(f.content_hash.clone());
            mtimes.push(f.mtime_nanos);
            chunk_idx.push(c.chunk_index);
            texts.push(c.text.clone());
            tokens.push(Some(c.token_count));
            context_ids.push(None);
            schema_versions.push(SCHEMA_VERSION as i32);
        }
    }

    let embedding_field = schema
        .field_with_name(ChunkColumn::EMBEDDING)
        .context("schema missing embedding column")?;
    let embedding: ArrayRef = new_null_array(embedding_field.data_type(), n);

    let cols: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(ids)),
        Arc::new(StringArray::from(collections)),
        Arc::new(StringArray::from(uris)),
        Arc::new(StringArray::from(paths)),
        Arc::new(StringArray::from(hashes)),
        Arc::new(Int64Array::from(mtimes)),
        Arc::new(Int32Array::from(chunk_idx)),
        Arc::new(StringArray::from(texts)),
        Arc::new(Int32Array::from(tokens)),
        embedding,
        Arc::new(StringArray::from(context_ids)),
        Arc::new(Int32Array::from(schema_versions)),
    ];

    RecordBatch::try_new(schema.clone(), cols).context("build chunk RecordBatch")
}
