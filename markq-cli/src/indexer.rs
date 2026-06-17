//! Indexer: walk a path, chunk markdown, write rows to the index.
//!
//! No embeddings yet — `embedding` is left null; `markq embed` fills it in
//! later. Indexing is not yet incremental (skip unchanged files, tombstone
//! removed ones); for now every run is a full append.

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

/// Outcome of an `markq index <path>` invocation. For now this reports the
/// totals straight; incremental indexing will extend it with `skipped` /
/// `tombstoned` counts.
pub struct IndexReport {
    pub files: usize,
    pub chunks: usize,
}

pub async fn run_index<I: Index>(idx: &I, root: &Path) -> Result<IndexReport> {
    let root = root
        .canonicalize()
        .with_context(|| format!("canonicalize {}", root.display()))?;

    let schema = idx.arrow_schema();
    let mut total_files = 0usize;
    let mut total_chunks = 0usize;
    let mut row_files: Vec<FileRows> = Vec::new();

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
        match build_file_rows(path, &root) {
            Ok(rows) => {
                total_files += 1;
                total_chunks += rows.chunks.len();
                row_files.push(rows);
            }
            Err(e) => warn!(file = %path.display(), error = %e, "skipping file"),
        }
    }

    if row_files.is_empty() {
        info!(path = %root.display(), "no markdown files found");
        return Ok(IndexReport {
            files: 0,
            chunks: 0,
        });
    }

    let batch = build_record_batch(&schema, &row_files)?;
    debug!(rows = batch.num_rows(), "upserting batch");
    idx.upsert_chunks(vec![batch])
        .await
        .context("upsert chunks into index")?;

    Ok(IndexReport {
        files: total_files,
        chunks: total_chunks,
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

fn build_file_rows(path: &Path, root: &Path) -> Result<FileRows> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes).into_owned();

    let content_hash = blake3::hash(&bytes).to_hex().to_string();
    let mtime_nanos = file_mtime_nanos(path)?;

    let rel = path.strip_prefix(root).unwrap_or(path);
    let rel_str = rel.to_string_lossy().replace('\\', "/");
    let uri = format!("markq://{DEFAULT_COLLECTION}/{rel_str}");
    let path_str = path.to_string_lossy().into_owned();

    let opts = ChunkOptions::default();
    let chunks = chunk_markdown(&text, &opts, &ApproxTokenizer);

    let chunk_rows = chunks
        .into_iter()
        .map(|c| {
            // Identity = (uri, content_hash, chunk_index). Two distinct files
            // with identical content (duplicated READMEs in a monorepo,
            // vendored notes, generated docs) must not collide on id; once
            // upsert becomes a real merge-on-id, a collision would let one
            // file overwrite another's chunk text.
            let mut h = blake3::Hasher::new();
            h.update(uri.as_bytes());
            h.update(&[0u8]); // separator so prefix collisions are impossible
            h.update(content_hash.as_bytes());
            h.update(&(c.index as u32).to_le_bytes());
            ChunkRow {
                id: h.finalize().to_hex().to_string(),
                chunk_index: c.index as i32,
                text: c.text,
                token_count: c.token_count.min(i32::MAX as usize) as i32,
            }
        })
        .collect();

    Ok(FileRows {
        path_str,
        uri,
        content_hash,
        mtime_nanos,
        chunks: chunk_rows,
    })
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
