//! LanceDB-backed `Index` implementation. covers open/create + the
//! metadata write that PHASE1_FOLLOWUPS.md item #4 calls for; query methods
//! are wired in (BM25), (vector), (hybrid).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int32Type};
use arrow_array::{Array, RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use futures::TryStreamExt;
use lance_index::scalar::FullTextSearchQuery;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::index::Index as LanceIndexKind;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::{Connection, Table};
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
    conn: Connection,
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

    /// Same as `open_or_create` but lets callers (e.g. tests) override the
    /// embedding dim. Production code should use the default until 
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
            conn,
            table,
            path: dataset_path.to_path_buf(),
            schema,
        })
    }

    /// Path the dataset lives at on disk.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The underlying `lancedb::Connection`. Exposed for ad-hoc admin work
    /// (e.g. `markq compact`) without leaking the rest of the table API.
    pub fn connection(&self) -> &Connection {
        &self.conn
    }
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
        let chunk_index = batch
            .column_by_name(ChunkColumn::CHUNK_INDEX)
            .context("missing chunk_index column")
            .map_err(Error::Backend)?
            .as_primitive::<Int32Type>();
        let text = column_string(&batch, ChunkColumn::TEXT)?;
        let score = batch
            .column_by_name("_score")
            .context("missing _score column")
            .map_err(Error::Backend)?
            .as_primitive::<Float32Type>();

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

    let schema_version: u32 = require(config, KEY_SCHEMA_VERSION)?
        .parse()
        .map_err(|_| Error::MetadataMissingKey(KEY_SCHEMA_VERSION))?;
    let lance_manifest_version: u64 = require(config, KEY_LANCE_MANIFEST_VERSION)?
        .parse()
        .map_err(|_| Error::MetadataMissingKey(KEY_LANCE_MANIFEST_VERSION))?;
    let lance_file_format_version = require(config, KEY_LANCE_FILE_FORMAT_VERSION)?.to_string();
    let lancedb_crate_version = require(config, KEY_LANCEDB_CRATE_VERSION)?.to_string();
    let embedder_model = config.get(KEY_EMBEDDER_MODEL).cloned();
    let embedder_dim = config
        .get(KEY_EMBEDDER_DIM)
        .and_then(|s| s.parse::<u32>().ok());

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
        // the index in sync without separate bookkeeping. will
        // gate this behind a "rebuild if fragmentation > N%" check.
        ensure_fts_index(&self.table).await?;
        Ok(())
    }

    async fn delete_by_path(&self, path: &str) -> Result<u64> {
        // wires the actual incremental-reindex tombstone path. For
        // this is enough to satisfy the trait surface and the
        // contract test.
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
        // hybrid path in ; for we pass the raw query through
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
        _embedding: &[f32],
        _k: usize,
        _collection: Option<&str>,
    ) -> Result<Vec<ChunkHit>> {
        // Vector retrieval lands in .
        Ok(Vec::new())
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
        // wires the OptimizeAction. no-op so `markq compact`
        // exits cleanly during smoke tests.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
