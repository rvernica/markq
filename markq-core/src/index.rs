//! The `Index` trait. Backend-agnostic by design — ships the LanceDB
//! implementation; the trait shape stays backend-agnostic so a fallback can
//! be re-added later if LanceDB's sub-1.0 API churn forces a swap, without
//! reshaping callers.
//!
//! Methods cover the v1 surface; v1.5 features (multi-collection filter
//! pushdown, context-tree joins, MCP `lex`/`vec`/`hyde` sub-queries) layer on
//! top of these primitives without changing the trait shape.

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use async_trait::async_trait;

use crate::error::Result;
use crate::metadata::DatasetMetadata;

/// A single retrieval hit. Backends populate score per their algorithm
/// (BM25 returns log-domain scores; vector returns cosine; hybrid returns
/// the fused RRF score).
#[derive(Debug, Clone)]
pub struct ChunkHit {
    pub id: String,
    pub path: String,
    pub uri: String,
    pub chunk_index: i32,
    pub text: String,
    pub score: f32,
}

/// Backend-agnostic index contract.
///
/// All retrieval methods take `collection: Option<&str>` so v1.5 multi-
/// collection filtering pushes down through the same call shape — 
/// just starts threading non-`None` values from the CLI.
#[async_trait]
pub trait Index: Send + Sync {
    /// Return the dataset's Arrow schema. Used by `markq inspect` and by
    /// schema round-trip tests.
    fn arrow_schema(&self) -> SchemaRef;

    /// Logical row count (Lance counts tombstones separately; this excludes
    /// them).
    async fn count_rows(&self) -> Result<usize>;

    /// Read the dataset metadata recorded at creation time (schema_version,
    /// Lance versions, optional embedder fields).
    async fn metadata(&self) -> Result<DatasetMetadata>;

    /// Append/upsert a batch of chunks. wires this to incremental
    /// reindex; just appends.
    ///
    /// **Cost note**: backends are permitted to do per-call index
    /// maintenance (e.g. the LanceDB backend rebuilds the FTS index on the
    /// `text` column with `.replace(true)`), so the cost can be O(rows) per
    /// invocation. Callers SHOULD batch a logical indexing run into a
    /// single `upsert_chunks` call rather than calling it once per file or
    /// per small batch. A future `finalize`/`compact` split can move the
    /// index rebuild out of the hot path; until then, batch up-front.
    async fn upsert_chunks(&self, batches: Vec<RecordBatch>) -> Result<()>;

    /// Tombstone all chunks for a source path. wires deletion;
    /// returns Ok with a no-op when the row count is zero so the
    /// trait surface compiles end-to-end.
    async fn delete_by_path(&self, path: &str) -> Result<u64>;

    /// BM25 retrieval over the `text` column. `collection` is reserved for
    /// filter pushdown.
    async fn bm25(&self, query: &str, k: usize, collection: Option<&str>) -> Result<Vec<ChunkHit>>;

    /// Vector KNN retrieval. Embedding dim must match the dataset's recorded
    /// dim (`metadata().embedder_dim`).
    async fn vector(
        &self,
        embedding: &[f32],
        k: usize,
        collection: Option<&str>,
    ) -> Result<Vec<ChunkHit>>;

    /// Hybrid retrieval: BM25 + vector in a single call, returned as two
    /// pre-fusion ranked lists. RRF fuses these in markq-core.
    async fn hybrid(
        &self,
        query: &str,
        embedding: &[f32],
        k: usize,
        collection: Option<&str>,
    ) -> Result<(Vec<ChunkHit>, Vec<ChunkHit>)>;

    /// Reclaim space from tombstoned rows and rebuild fragmented indexes.
    /// wires the threshold logic.
    async fn compact(&self) -> Result<()>;
}
