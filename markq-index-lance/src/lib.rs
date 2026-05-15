//! LanceDB-backed `Index` implementation. Phase 1 covers open/create + the
//! metadata write that PHASE1_FOLLOWUPS.md item #4 calls for; query methods
//! are wired in Phase 3 (BM25), Phase 4 (vector), Phase 5 (hybrid).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use arrow_array::{
    Array, Float32Array, Int32Array, RecordBatch, RecordBatchIterator, RecordBatchReader,
    StringArray,
};
use arrow_schema::{DataType, SchemaRef};
use async_trait::async_trait;
use futures::TryStreamExt;
use lance_index::scalar::FullTextSearchQuery;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::index::vector::IvfHnswSqIndexBuilder;
use lancedb::index::Index as LanceIndexKind;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::{DistanceType, Table};
use markq_core::{
    chunk_arrow_schema, ChunkColumn, ChunkHit, DatasetMetadata, Error, Index, Result,
    EMBEDDING_DIM_DEFAULT, KEY_EMBEDDER_DIM, KEY_EMBEDDER_MODEL, KEY_LANCEDB_CRATE_VERSION,
    KEY_LANCE_FILE_FORMAT_VERSION, KEY_LANCE_MANIFEST_VERSION, KEY_SCHEMA_VERSION, SCHEMA_VERSION,
};
use tracing::{debug, info};

/// Pinned version of the `lancedb` crate. Mirrors the `=0.27.2` exact pin in
/// the workspace `Cargo.toml`. If you bump the pin, bump this constant in
/// the same commit — `markq doctor` will flag the divergence.
pub const LANCEDB_CRATE_VERSION: &str = "0.27.2";

pub struct LanceIndex {
    table: Table,
    /// The dataset directory (e.g. `~/.markq/chunks.lance`). Held so
    /// `markq inspect` can print it.
    path: PathBuf,
    schema: SchemaRef,
}

impl LanceIndex {
    /// Open an existing dataset or create one with markq's chunk schema and
    /// metadata baked in.
    pub async fn open_or_create(dataset_path: &Path) -> Result<Self> {
        Self::open_or_create_with_dim(dataset_path, EMBEDDING_DIM_DEFAULT).await
    }

    /// Open an existing dataset. Errors with a clean "dataset not found"
    /// message if the path doesn't already point at a markq dataset, so
    /// read-only commands like `markq search` / `markq vsearch` don't
    /// silently materialize empty datasets on typos.
    pub async fn open(dataset_path: &Path) -> Result<Self> {
        if !dataset_path.exists() {
            return Err(Error::Backend(anyhow::anyhow!(
                "dataset not found at {} (run `markq index <path>` first)",
                dataset_path.display()
            )));
        }
        Self::open_or_create_with_dim(dataset_path, EMBEDDING_DIM_DEFAULT).await
    }

    /// Same as `open_or_create` but lets callers (e.g. tests) override the
    /// embedding dim. Production code should use the default until Phase 4
    /// wires per-embedder dim into config.
    pub async fn open_or_create_with_dim(dataset_path: &Path, embedding_dim: i32) -> Result<Self> {
        // LanceDB's `connect(uri)` opens a *database directory* that holds
        // many tables; each table lives at `<uri>/<table_name>.lance/`. We
        // treat the user-facing `dataset_path` as the table directory itself
        // — connect to its parent, take the file stem as the table name —
        // so `~/.markq/chunks.lance` is *the* dataset on disk, not nested
        // one level deeper.
        let parent = dataset_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .context("dataset path must have a parent directory")
            .map_err(Error::Backend)?;
        let table_name = dataset_path
            .file_stem()
            .and_then(|s| s.to_str())
            .context("dataset path must end in `<name>.lance`")
            .map_err(Error::Backend)?
            .to_string();

        std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        let db_uri = parent
            .to_str()
            .context("dataset parent path is not valid UTF-8")
            .map_err(Error::Backend)?;

        let conn = lancedb::connect(db_uri)
            .execute()
            .await
            .context("connect to lancedb")
            .map_err(Error::Backend)?;

        let names = conn
            .table_names()
            .execute()
            .await
            .context("list table names")
            .map_err(Error::Backend)?;

        let schema = chunk_arrow_schema(embedding_dim);

        let table = if names.iter().any(|n| n == &table_name) {
            debug!(path = %dataset_path.display(), "opening existing chunks table");
            conn.open_table(&table_name)
                .execute()
                .await
                .context("open chunks table")
                .map_err(Error::Backend)?
        } else {
            info!(path = %dataset_path.display(), "creating chunks table");
            let empty: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
                std::iter::empty::<std::result::Result<RecordBatch, arrow::error::ArrowError>>(),
                schema.clone(),
            ));
            let table = conn
                .create_table(&table_name, empty)
                .execute()
                .await
                .context("create chunks table")
                .map_err(Error::Backend)?;
            write_initial_metadata(&table).await?;
            table
        };

        Ok(LanceIndex {
            table,
            path: dataset_path.to_path_buf(),
            schema,
        })
    }

    /// Path the dataset lives at on disk.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// On first embed, record `embedder_model` + `embedder_dim` in the
    /// dataset's user metadata. On subsequent calls, verify the existing
    /// values match — a mismatch is `Error::EmbedderDimMismatch` (dim) or a
    /// `Error::Backend` carrying a clean message (model id).
    ///
    /// Returns `true` if this call wrote the metadata (first time), `false`
    /// if validation against existing metadata succeeded.
    pub async fn validate_or_record_embedder(&self, model_id: &str, dim: u32) -> Result<bool> {
        let native = self
            .table
            .as_native()
            .context("native table required to update embedder metadata")
            .map_err(Error::Backend)?;

        let manifest = native
            .manifest()
            .await
            .context("read manifest")
            .map_err(Error::Backend)?;
        let config = &manifest.config;

        match (
            config.get(KEY_EMBEDDER_MODEL),
            config.get(KEY_EMBEDDER_DIM),
        ) {
            (Some(existing_model), Some(existing_dim_raw)) => {
                let existing_dim: u32 = existing_dim_raw.parse().map_err(|source| {
                    Error::MetadataInvalidValue {
                        key: KEY_EMBEDDER_DIM,
                        value: existing_dim_raw.clone(),
                        source,
                    }
                })?;
                if existing_dim != dim {
                    return Err(Error::EmbedderDimMismatch {
                        dataset: existing_dim,
                        embedder: dim,
                    });
                }
                if existing_model != model_id {
                    return Err(Error::Backend(anyhow::anyhow!(
                        "embedder model mismatch: dataset built with {existing_model}, current embedder {model_id}"
                    )));
                }
                Ok(false)
            }
            (None, None) => {
                // Cross-check the caller's dim against the dataset's own
                // FixedSizeList width before persisting. Otherwise a future
                // KnownModel with a different dim would happily record a
                // mismatched value here and only fail later as an opaque
                // Arrow length error inside `apply_embeddings`.
                let schema_dim = embedding_dim_from_schema(&self.schema)?;
                if schema_dim != dim {
                    return Err(Error::EmbedderDimMismatch {
                        dataset: schema_dim,
                        embedder: dim,
                    });
                }
                native
                    .update_config(vec![
                        (KEY_EMBEDDER_MODEL.to_string(), model_id.to_string()),
                        (KEY_EMBEDDER_DIM.to_string(), dim.to_string()),
                    ])
                    .await
                    .context("write embedder metadata")
                    .map_err(Error::Backend)?;
                Ok(true)
            }
            _ => Err(Error::Backend(anyhow::anyhow!(
                "embedder metadata is partial — exactly one of {KEY_EMBEDDER_MODEL}/{KEY_EMBEDDER_DIM} is set; this should not happen"
            ))),
        }
    }

    /// Stream rows whose `embedding` column is null. Returns `(id, text)`
    /// pairs. The full RecordBatch is also returned so callers can pass it
    /// straight back through `merge_insert` after filling in `embedding` —
    /// avoiding a second scan.
    ///
    /// `limit` caps the rows returned in one call; `None` means "all
    /// remaining unembedded rows".
    pub async fn scan_unembedded(&self, limit: Option<usize>) -> Result<Vec<RecordBatch>> {
        let mut q = self
            .table
            .query()
            .only_if(format!("{} IS NULL", ChunkColumn::EMBEDDING));
        if let Some(l) = limit {
            q = q.limit(l);
        }
        let mut stream = q
            .execute()
            .await
            .context("execute unembedded scan")
            .map_err(Error::Backend)?;
        let mut out = Vec::new();
        while let Some(batch) = stream
            .try_next()
            .await
            .context("read unembedded batch")
            .map_err(Error::Backend)?
        {
            out.push(batch);
        }
        Ok(out)
    }

    /// Merge updated rows (same `id`, now with `embedding` populated) back
    /// into the table.
    ///
    /// Note: this does **not** rebuild the vector index — callers running a
    /// multi-batch loop should call [`LanceIndex::rebuild_vector_index`] once
    /// after the loop instead of paying for a full index rebuild per batch.
    pub async fn apply_embeddings(&self, updated: Vec<RecordBatch>) -> Result<u64> {
        if updated.is_empty() {
            return Ok(0);
        }
        let row_count: u64 = updated.iter().map(|b| b.num_rows() as u64).sum();
        let schema = self.schema.clone();
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
            updated.into_iter().map(Ok),
            schema,
        ));
        let mut merge = self.table.merge_insert(&[ChunkColumn::ID]);
        merge.when_matched_update_all(None);
        merge
            .execute(reader)
            .await
            .context("merge_insert embeddings")
            .map_err(Error::Backend)?;
        Ok(row_count)
    }

    /// (Re)build the HNSW Cosine vector index on the `embedding` column.
    /// Idempotent: Lance treats `create_index` as create-or-replace.
    pub async fn rebuild_vector_index(&self) -> Result<()> {
        ensure_vector_index(&self.table).await
    }
}

/// Read the `embedding` column's `FixedSizeList` element count from the
/// dataset's Arrow schema. The column is locked in at create time, so this
/// is the dim a freshly recorded `markq.embedder_dim` must match.
fn embedding_dim_from_schema(schema: &SchemaRef) -> Result<u32> {
    let field = schema
        .field_with_name(ChunkColumn::EMBEDDING)
        .context("schema missing embedding column")
        .map_err(Error::Backend)?;
    match field.data_type() {
        DataType::FixedSizeList(_, n) => Ok(*n as u32),
        other => Err(Error::Backend(anyhow::anyhow!(
            "embedding column has unexpected data type {other:?}; expected FixedSizeList"
        ))),
    }
}

async fn ensure_vector_index(table: &Table) -> Result<()> {
    // IvfHnswSq with Cosine matches the `vector()` query path. Like the FTS
    // path, we create-or-replace per call; Phase 8 will gate behind
    // fragmentation thresholds. Defaults (m=20, ef_construction=300) match
    // upstream Lance defaults; only the distance type is overridden.
    let builder = IvfHnswSqIndexBuilder::default().distance_type(DistanceType::Cosine);
    table
        .create_index(
            &[ChunkColumn::EMBEDDING],
            LanceIndexKind::IvfHnswSq(builder),
        )
        .replace(true)
        .execute()
        .await
        .context("create vector index on embedding")
        .map_err(Error::Backend)?;
    Ok(())
}

async fn write_initial_metadata(table: &Table) -> Result<()> {
    // `update_config` (user metadata) lives on `NativeTable`. Local datasets
    // are always native — `as_native()` returns `None` only for the remote
    // (LanceDB Cloud) flavor, which we don't use.
    let native = table
        .as_native()
        .context("expected native (local) lancedb table; remote not supported in v1")
        .map_err(Error::Backend)?;

    let manifest = native
        .manifest()
        .await
        .context("read manifest")
        .map_err(Error::Backend)?;

    let entries: Vec<(String, String)> = vec![
        (KEY_SCHEMA_VERSION.to_string(), SCHEMA_VERSION.to_string()),
        (
            KEY_LANCE_MANIFEST_VERSION.to_string(),
            manifest.version.to_string(),
        ),
        (
            KEY_LANCE_FILE_FORMAT_VERSION.to_string(),
            manifest.data_storage_format.version.clone(),
        ),
        (
            KEY_LANCEDB_CRATE_VERSION.to_string(),
            LANCEDB_CRATE_VERSION.to_string(),
        ),
    ];

    native
        .update_config(entries)
        .await
        .context("write dataset metadata")
        .map_err(Error::Backend)?;
    Ok(())
}

/// Build (or replace) the BM25 FTS index on the `text` column. Tokenizer
/// settings mirror the spike 0c configuration that achieved 0.99 overlap@10
/// against the qmd reference: `simple` base + lower-case + Porter stem +
/// ASCII folding + stop-words preserved.
async fn ensure_fts_index(table: &Table) -> Result<()> {
    let params = FtsIndexBuilder::default()
        .base_tokenizer("simple".to_string())
        .lower_case(true)
        .stem(true)
        .remove_stop_words(false)
        .ascii_folding(true);
    table
        .create_index(&[ChunkColumn::TEXT], LanceIndexKind::FTS(params))
        .replace(true)
        .execute()
        .await
        .context("create FTS index on text")
        .map_err(Error::Backend)?;
    Ok(())
}

async fn bm25_search(table: &Table, query: &str, k: usize) -> Result<Vec<ChunkHit>> {
    let mut stream = table
        .query()
        .full_text_search(FullTextSearchQuery::new(query.to_string()))
        .select(Select::Columns(vec![
            ChunkColumn::ID.to_string(),
            ChunkColumn::PATH.to_string(),
            ChunkColumn::URI.to_string(),
            ChunkColumn::CHUNK_INDEX.to_string(),
            ChunkColumn::TEXT.to_string(),
            "_score".to_string(),
        ]))
        .limit(k)
        .execute()
        .await
        .context("execute full_text_search")
        .map_err(Error::Backend)?;

    let mut hits = Vec::new();
    while let Some(batch) = stream
        .try_next()
        .await
        .context("read fts result batch")
        .map_err(Error::Backend)?
    {
        let id = column_string(&batch, ChunkColumn::ID)?;
        let path = column_string(&batch, ChunkColumn::PATH)?;
        let uri = column_string(&batch, ChunkColumn::URI)?;
        let chunk_index = column_int32(&batch, ChunkColumn::CHUNK_INDEX)?;
        let text = column_string(&batch, ChunkColumn::TEXT)?;
        // `_score` is an implicit column produced by Lance's FTS path; its
        // numeric type is an upstream contract that may shift on a lancedb
        // bump. Surface a type mismatch as a clean Error::Backend rather
        // than the panic that `as_primitive` would raise.
        let score = column_float32(&batch, "_score")?;

        for i in 0..batch.num_rows() {
            hits.push(ChunkHit {
                id: id.value(i).to_string(),
                path: path.value(i).to_string(),
                uri: uri.value(i).to_string(),
                chunk_index: chunk_index.value(i),
                text: text.value(i).to_string(),
                score: score.value(i),
            });
        }
    }
    Ok(hits)
}

async fn vector_search(table: &Table, embedding: &[f32], k: usize) -> Result<Vec<ChunkHit>> {
    let query = table
        .query()
        .nearest_to(embedding.to_vec())
        .context("nearest_to(query vector)")
        .map_err(Error::Backend)?
        .distance_type(DistanceType::Cosine)
        .select(Select::Columns(vec![
            ChunkColumn::ID.to_string(),
            ChunkColumn::PATH.to_string(),
            ChunkColumn::URI.to_string(),
            ChunkColumn::CHUNK_INDEX.to_string(),
            ChunkColumn::TEXT.to_string(),
            "_distance".to_string(),
        ]))
        .limit(k);

    let mut stream = query
        .execute()
        .await
        .context("execute vector query")
        .map_err(Error::Backend)?;

    let mut hits = Vec::new();
    while let Some(batch) = stream
        .try_next()
        .await
        .context("read vector result batch")
        .map_err(Error::Backend)?
    {
        let id = column_string(&batch, ChunkColumn::ID)?;
        let path = column_string(&batch, ChunkColumn::PATH)?;
        let uri = column_string(&batch, ChunkColumn::URI)?;
        let chunk_index = column_int32(&batch, ChunkColumn::CHUNK_INDEX)?;
        let text = column_string(&batch, ChunkColumn::TEXT)?;
        // Lance emits `_distance` for vector queries; cosine distance is in
        // [0, 2]. Convert to a similarity in [-1, 1] (`1 - distance`) so
        // higher-is-better matches the BM25 convention and downstream RRF
        // (Phase 5) can treat scores uniformly.
        let dist = column_float32(&batch, "_distance")?;

        for i in 0..batch.num_rows() {
            hits.push(ChunkHit {
                id: id.value(i).to_string(),
                path: path.value(i).to_string(),
                uri: uri.value(i).to_string(),
                chunk_index: chunk_index.value(i),
                text: text.value(i).to_string(),
                score: 1.0 - dist.value(i),
            });
        }
    }
    Ok(hits)
}

fn column_string<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    let arr = batch
        .column_by_name(name)
        .with_context(|| format!("missing {name} column"))
        .map_err(Error::Backend)?;
    arr.as_any()
        .downcast_ref::<StringArray>()
        .with_context(|| format!("{name} column is not utf8"))
        .map_err(Error::Backend)
}

fn column_int32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int32Array> {
    let arr = batch
        .column_by_name(name)
        .with_context(|| format!("missing {name} column"))
        .map_err(Error::Backend)?;
    arr.as_any()
        .downcast_ref::<Int32Array>()
        .with_context(|| format!("{name} column is not int32"))
        .map_err(Error::Backend)
}

fn column_float32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float32Array> {
    let arr = batch
        .column_by_name(name)
        .with_context(|| format!("missing {name} column"))
        .map_err(Error::Backend)?;
    arr.as_any()
        .downcast_ref::<Float32Array>()
        .with_context(|| format!("{name} column is not float32"))
        .map_err(Error::Backend)
}

async fn read_metadata(table: &Table) -> Result<DatasetMetadata> {
    let native = table
        .as_native()
        .context("native table required to read user metadata")
        .map_err(Error::Backend)?;

    let manifest = native
        .manifest()
        .await
        .context("read manifest")
        .map_err(Error::Backend)?;

    // Lance's user-defined config is exposed via the manifest's `config` map.
    let config: &HashMap<String, String> = &manifest.config;

    fn require<'a>(c: &'a HashMap<String, String>, key: &'static str) -> Result<&'a str> {
        c.get(key)
            .map(String::as_str)
            .ok_or(Error::MetadataMissingKey(key))
    }

    fn parse_int<T: std::str::FromStr<Err = std::num::ParseIntError>>(
        raw: &str,
        key: &'static str,
    ) -> Result<T> {
        raw.parse().map_err(|source| Error::MetadataInvalidValue {
            key,
            value: raw.to_string(),
            source,
        })
    }

    let schema_version: u32 = parse_int(require(config, KEY_SCHEMA_VERSION)?, KEY_SCHEMA_VERSION)?;
    let lance_manifest_version: u64 = parse_int(
        require(config, KEY_LANCE_MANIFEST_VERSION)?,
        KEY_LANCE_MANIFEST_VERSION,
    )?;
    let lance_file_format_version = require(config, KEY_LANCE_FILE_FORMAT_VERSION)?.to_string();
    let lancedb_crate_version = require(config, KEY_LANCEDB_CRATE_VERSION)?.to_string();
    let embedder_model = config.get(KEY_EMBEDDER_MODEL).cloned();
    let embedder_dim = match config.get(KEY_EMBEDDER_DIM) {
        Some(raw) => Some(parse_int(raw, KEY_EMBEDDER_DIM)?),
        None => None,
    };

    Ok(DatasetMetadata {
        schema_version,
        lance_manifest_version,
        lance_file_format_version,
        lancedb_crate_version,
        embedder_model,
        embedder_dim,
    })
}

#[async_trait]
impl Index for LanceIndex {
    fn arrow_schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    async fn count_rows(&self) -> Result<usize> {
        self.table
            .count_rows(None)
            .await
            .context("count rows")
            .map_err(Error::Backend)
    }

    async fn metadata(&self) -> Result<DatasetMetadata> {
        read_metadata(&self.table).await
    }

    async fn upsert_chunks(&self, batches: Vec<RecordBatch>) -> Result<()> {
        if batches.is_empty() {
            return Ok(());
        }
        let schema = self.schema.clone();
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
            batches.into_iter().map(Ok),
            schema,
        ));
        self.table
            .add(reader)
            .execute()
            .await
            .context("append chunks")
            .map_err(Error::Backend)?;
        // Build / refresh the BM25 FTS index on `text`. Lance treats
        // `create_index` as create-or-replace, so calling per upsert keeps
        // the index in sync without separate bookkeeping. Phase 8 will
        // gate this behind a "rebuild if fragmentation > N%" check.
        ensure_fts_index(&self.table).await?;
        Ok(())
    }

    async fn delete_by_path(&self, path: &str) -> Result<u64> {
        // Phase 8 wires the actual incremental-reindex tombstone path. For
        // Phase 1 this is enough to satisfy the trait surface and the
        // contract test.
        //
        // SAFETY ASSUMPTION: `path` is a filesystem path produced by markq's
        // own indexer (canonicalized; no control bytes). The hand-rolled
        // single-quote doubling matches Lance/DataFusion's expression-parser
        // escape rules, but it is not a general SQL sanitizer. If a future
        // caller ever threads user-controlled strings through this method
        // (e.g. a `markq delete <pattern>` command), swap to a parameter-
        // bound delete API rather than extending this escape.
        let escaped = path.replace('\'', "''");
        let predicate = format!("path = '{escaped}'");
        let res = self
            .table
            .delete(&predicate)
            .await
            .context("delete by path")
            .map_err(Error::Backend)?;
        Ok(res.num_deleted_rows)
    }

    async fn bm25(
        &self,
        query: &str,
        k: usize,
        _collection: Option<&str>,
    ) -> Result<Vec<ChunkHit>> {
        // PHASE1_FOLLOWUPS #2 (hyphen-aware FTS5 sanitizer) lands with the
        // hybrid path in Phase 5; for Phase 3 we pass the raw query through
        // and let LanceDB's tokenizer match document-side terms. The 0c
        // spike showed this recalls hyphenated identifiers correctly on the
        // lance side — the regression was specifically on the qmd / SQLite
        // FTS5 side, which we don't ship.
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        // No rows → no FTS index has been built yet. Short-circuit so the
        // contract test (which calls bm25 on a fresh dataset) doesn't trip
        // a "missing index" error from Lance.
        if self.count_rows().await? == 0 {
            return Ok(Vec::new());
        }
        bm25_search(&self.table, query, k).await
    }

    async fn vector(
        &self,
        embedding: &[f32],
        k: usize,
        _collection: Option<&str>,
    ) -> Result<Vec<ChunkHit>> {
        if embedding.is_empty() {
            return Ok(Vec::new());
        }
        if self.count_rows().await? == 0 {
            return Ok(Vec::new());
        }
        vector_search(&self.table, embedding, k).await
    }

    async fn hybrid(
        &self,
        _query: &str,
        _embedding: &[f32],
        _k: usize,
        _collection: Option<&str>,
    ) -> Result<(Vec<ChunkHit>, Vec<ChunkHit>)> {
        Ok((Vec::new(), Vec::new()))
    }

    async fn compact(&self) -> Result<()> {
        // Phase 8 wires the OptimizeAction. Phase 1: no-op so `markq compact`
        // exits cleanly during smoke tests.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::{
        new_null_array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, Int64Array,
        StringArray,
    };
    use arrow_schema::{DataType, Field};

    #[tokio::test]
    async fn validate_or_record_embedder_first_then_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chunks.lance");
        // Build with a small explicit dim so the round-trip test doesn't
        // need a 1024-wide allocation.
        let idx = LanceIndex::open_or_create_with_dim(&path, 8).await.unwrap();

        // First call records the metadata.
        let wrote = idx
            .validate_or_record_embedder("test/embedder-A", 8)
            .await
            .unwrap();
        assert!(wrote, "first call should write metadata");

        // Same model + dim → ok, no-op.
        let wrote2 = idx
            .validate_or_record_embedder("test/embedder-A", 8)
            .await
            .unwrap();
        assert!(!wrote2, "second call with matching args should be a no-op");

        // Different dim → typed dim-mismatch error.
        let err = idx
            .validate_or_record_embedder("test/embedder-A", 16)
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                Error::EmbedderDimMismatch {
                    dataset: 8,
                    embedder: 16
                }
            ),
            "expected EmbedderDimMismatch, got: {err:?}"
        );

        // Same dim, different model id → Error::Backend with a clean message
        // (a separate code path from EmbedderDimMismatch — silently accepting
        // would let a Mean-pooled and a Last-pooled embedder co-mingle).
        let err = idx
            .validate_or_record_embedder("test/embedder-B", 8)
            .await
            .unwrap_err();
        match err {
            Error::Backend(e) => {
                let msg = format!("{e}");
                assert!(
                    msg.contains("embedder model mismatch"),
                    "expected model-mismatch message, got: {msg}"
                );
                assert!(msg.contains("test/embedder-A"));
                assert!(msg.contains("test/embedder-B"));
            }
            other => panic!("expected Error::Backend for model-id mismatch, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn validate_or_record_embedder_rejects_schema_dim_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chunks.lance");
        // Dataset's FixedSizeList width is 4; recording with embedder dim=8
        // must surface a clean EmbedderDimMismatch rather than write the
        // wrong dim and fail later inside merge_insert.
        let idx = LanceIndex::open_or_create_with_dim(&path, 4).await.unwrap();
        let err = idx
            .validate_or_record_embedder("test/embedder-x", 8)
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                Error::EmbedderDimMismatch {
                    dataset: 4,
                    embedder: 8
                }
            ),
            "expected EmbedderDimMismatch{{dataset:4, embedder:8}}, got: {err:?}"
        );
        // And no metadata leaked into the dataset on the failed call.
        let md = idx.metadata().await.unwrap();
        assert!(md.embedder_model.is_none());
        assert!(md.embedder_dim.is_none());
    }

    /// Build a single-row batch with explicit f32 embedding, merge it into a
    /// fresh dataset, and confirm `vector()` returns the row with a cosine
    /// similarity in the expected direction.
    #[tokio::test]
    async fn vector_round_trip_small_dim() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chunks.lance");
        let idx = LanceIndex::open_or_create_with_dim(&path, 4).await.unwrap();
        let schema = idx.arrow_schema();

        // Two rows with orthogonal-ish unit-length embeddings.
        let ids = StringArray::from(vec!["a", "b"]);
        let collections = StringArray::from(vec!["default", "default"]);
        let uris = StringArray::from(vec!["markq://default/a.md", "markq://default/b.md"]);
        let paths = StringArray::from(vec!["a.md", "b.md"]);
        let hashes = StringArray::from(vec!["hash-a", "hash-b"]);
        let mtimes = Int64Array::from(vec![0i64, 0]);
        let chunk_idx = Int32Array::from(vec![0i32, 0]);
        let texts = StringArray::from(vec!["alpha", "bravo"]);
        let tokens = Int32Array::from(vec![Some(1i32), Some(1)]);
        let ctx_ids: arrow_array::ArrayRef =
            Arc::new(StringArray::from(vec![None as Option<&str>, None]));
        let schema_versions = Int32Array::from(vec![SCHEMA_VERSION as i32; 2]);

        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Float32Array::from(vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let embedding = FixedSizeListArray::try_new(item, 4, Arc::new(values), None).unwrap();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids),
                Arc::new(collections),
                Arc::new(uris),
                Arc::new(paths),
                Arc::new(hashes),
                Arc::new(mtimes),
                Arc::new(chunk_idx),
                Arc::new(texts),
                Arc::new(tokens),
                Arc::new(embedding),
                ctx_ids,
                Arc::new(schema_versions),
            ],
        )
        .unwrap();

        // First upsert: rows land with embedding populated (the column is
        // nullable but we're filling it).
        idx.upsert_chunks(vec![batch]).await.unwrap();
        // Manually build the vector index since upsert path covers FTS only.
        ensure_vector_index(&idx.table).await.unwrap();

        // Query with a vector close to row "a" — expect "a" first.
        let hits = idx.vector(&[0.9, 0.1, 0.0, 0.0], 2, None).await.unwrap();
        assert!(!hits.is_empty(), "vector() returned no hits");
        assert_eq!(hits[0].id, "a", "expected 'a' as nearest, got {hits:?}");
        // Cosine similarity = 1 - cosine_distance; identical-direction
        // vectors score ~1, orthogonal ~0. We're not quite identical so
        // just demand the right ordering and a positive score.
        assert!(hits[0].score > 0.0);
    }

    /// Build the non-embedding columns of a chunk batch. The caller supplies
    /// the `embedding` column; this helper handles every other deterministic
    /// field so the two batches (NULL-embedding seed vs populated update)
    /// share construction logic.
    fn chunk_cols_without_embedding(ids: &[&str]) -> Vec<ArrayRef> {
        let n = ids.len();
        let uris: Vec<String> = ids.iter().map(|i| format!("markq://default/{i}")).collect();
        let paths: Vec<String> = ids.iter().map(|i| format!("{i}.md")).collect();
        let hashes: Vec<String> = ids.iter().map(|i| format!("hash-{i}")).collect();
        let texts: Vec<String> = ids.iter().map(|i| format!("text for {i}")).collect();
        vec![
            Arc::new(StringArray::from(ids.to_vec())),
            Arc::new(StringArray::from(vec!["default"; n])),
            Arc::new(StringArray::from(uris)),
            Arc::new(StringArray::from(paths)),
            Arc::new(StringArray::from(hashes)),
            Arc::new(Int64Array::from(vec![0i64; n])),
            Arc::new(Int32Array::from(vec![0i32; n])),
            Arc::new(StringArray::from(texts)),
            Arc::new(Int32Array::from(vec![Some(1i32); n])),
            // embedding slot — caller fills index 9
            Arc::new(StringArray::from(Vec::<&str>::new())), // placeholder, replaced below
            Arc::new(StringArray::from(vec![None as Option<&str>; n])),
            Arc::new(Int32Array::from(vec![SCHEMA_VERSION as i32; n])),
        ]
    }

    fn batch_with_embedding(schema: &SchemaRef, ids: &[&str], embedding: ArrayRef) -> RecordBatch {
        let mut cols = chunk_cols_without_embedding(ids);
        cols[9] = embedding;
        RecordBatch::try_new(schema.clone(), cols).unwrap()
    }

    #[tokio::test]
    async fn scan_unembedded_and_apply_embeddings_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chunks.lance");
        let idx = LanceIndex::open_or_create_with_dim(&path, 4).await.unwrap();
        let schema = idx.arrow_schema();

        // Seed 4 rows with NULL embeddings, mirroring how `markq index` lands
        // chunks before `markq embed` runs.
        let embedding_dtype = schema
            .field_with_name(ChunkColumn::EMBEDDING)
            .unwrap()
            .data_type();
        let null_embedding: ArrayRef = new_null_array(embedding_dtype, 4);
        let seed = batch_with_embedding(&schema, &["a", "b", "c", "d"], null_embedding);
        idx.upsert_chunks(vec![seed]).await.unwrap();
        assert_eq!(idx.count_rows().await.unwrap(), 4);

        // Initial scan returns all four unembedded rows.
        let unembedded = idx.scan_unembedded(None).await.unwrap();
        let total: usize = unembedded.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 4, "all rows should be unembedded initially");

        // Populate embeddings on two rows ("a" and "c") via merge_insert.
        // The merge is keyed on `id`, so the other columns must match
        // existing rows exactly to avoid creating new ones.
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Float32Array::from(vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let populated: ArrayRef =
            Arc::new(FixedSizeListArray::try_new(item, 4, Arc::new(values), None).unwrap());
        let updated = batch_with_embedding(&schema, &["a", "c"], populated);
        let n_applied = idx.apply_embeddings(vec![updated]).await.unwrap();
        assert_eq!(n_applied, 2);
        assert_eq!(
            idx.count_rows().await.unwrap(),
            4,
            "merge_insert must not create new rows when ids match"
        );

        // After the merge, only "b" and "d" should still come back as
        // unembedded.
        let remaining = idx.scan_unembedded(None).await.unwrap();
        let mut remaining_ids: Vec<String> = remaining
            .iter()
            .flat_map(|batch| {
                let ids = column_string(batch, ChunkColumn::ID).unwrap();
                (0..batch.num_rows())
                    .map(|i| ids.value(i).to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        remaining_ids.sort();
        assert_eq!(remaining_ids, vec!["b".to_string(), "d".to_string()]);
    }

    #[tokio::test]
    async fn create_writes_metadata_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chunks.lance");
        let idx = LanceIndex::open_or_create(&path).await.unwrap();
        assert_eq!(idx.count_rows().await.unwrap(), 0);

        let md = idx.metadata().await.unwrap();
        assert_eq!(md.schema_version, SCHEMA_VERSION);
        assert_eq!(md.lancedb_crate_version, LANCEDB_CRATE_VERSION);
        assert!(
            !md.lance_file_format_version.is_empty(),
            "Lance file format version must be recorded at create time"
        );
    }

    #[tokio::test]
    async fn reopen_preserves_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chunks.lance");
        let md_a = {
            let idx = LanceIndex::open_or_create(&path).await.unwrap();
            idx.metadata().await.unwrap()
        };
        let md_b = {
            let idx = LanceIndex::open_or_create(&path).await.unwrap();
            idx.metadata().await.unwrap()
        };
        assert_eq!(md_a.schema_version, md_b.schema_version);
        assert_eq!(
            md_a.lance_file_format_version,
            md_b.lance_file_format_version
        );
        assert_eq!(md_a.lancedb_crate_version, md_b.lancedb_crate_version);
    }
}
