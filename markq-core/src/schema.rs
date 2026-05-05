//! Arrow schema for the chunk table — designed once in to its final
//! v1.5 shape so adding `markq context add` and multi-collection
//! routing is additive rather than a Lance migration + full
//! re-embed.
//!
//! Ordering and nullability are load-bearing: `embedding`, `context_id`, and
//! `tokens` are nullable so chunks can be written before `markq embed`
//! runs.

use std::path::PathBuf;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};

/// Default embedding dimension for Qwen3-Embedding-0.6B (the v1 default).
/// Recorded in dataset metadata at first write so the embedder cannot silently
/// be swapped for a different-dim model.
pub const EMBEDDING_DIM_DEFAULT: i32 = 1024;

/// Symbolic names for chunk columns. Use these instead of bare strings so
/// renames are a compile error.
pub struct ChunkColumn;

impl ChunkColumn {
    pub const ID: &'static str = "id";
    pub const COLLECTION: &'static str = "collection";
    pub const URI: &'static str = "uri";
    pub const PATH: &'static str = "path";
    pub const CONTENT_HASH: &'static str = "content_hash";
    pub const MTIME: &'static str = "mtime";
    pub const CHUNK_INDEX: &'static str = "chunk_index";
    pub const TEXT: &'static str = "text";
    pub const TOKENS: &'static str = "tokens";
    pub const EMBEDDING: &'static str = "embedding";
    pub const CONTEXT_ID: &'static str = "context_id";
    pub const SCHEMA_VERSION: &'static str = "schema_version";
}

/// The chunk Arrow schema — final shape, columns reserved for v1.5 features.
pub fn chunk_arrow_schema(embedding_dim: i32) -> SchemaRef {
    Arc::new(Schema::new(vec![
        // Stable chunk identity. Hash of (content_hash, chunk_index) so the
        // same chunk in the same file across re-indexes has the same id.
        Field::new(ChunkColumn::ID, DataType::Utf8, false),
        // Routing column for `-c <name>` filter pushdown.
        // Always populated; defaults to "default" before multi-collection lands.
        Field::new(ChunkColumn::COLLECTION, DataType::Utf8, false),
        // Canonical URI of the source document (`markq://<collection>/<path>`).
        // Reserved here so context-tree prefix matching is a
        // pure-additive change rather than a schema migration.
        Field::new(ChunkColumn::URI, DataType::Utf8, false),
        // Filesystem path relative to the collection root.
        Field::new(ChunkColumn::PATH, DataType::Utf8, false),
        // Blake3 / sha256 hex digest of the source file. Drives incremental
        // reindex: unchanged hash → skip the file entirely.
        Field::new(ChunkColumn::CONTENT_HASH, DataType::Utf8, false),
        // Source mtime as nanoseconds since epoch (i64). Cheap pre-filter
        // before content_hash; mtime change without content change is a no-op.
        Field::new(ChunkColumn::MTIME, DataType::Int64, false),
        // 0-based chunk index within the source document.
        Field::new(ChunkColumn::CHUNK_INDEX, DataType::Int32, false),
        // The chunk's text. BM25 indexes this column.
        Field::new(ChunkColumn::TEXT, DataType::Utf8, false),
        // Token count for the chunk per the embedder tokenizer. Nullable for
        // rows written before `markq embed` (which fills it in alongside the
        // embedding).
        Field::new(ChunkColumn::TOKENS, DataType::Int32, true),
        // Vector embedding. Nullable until . Fixed-size list keeps
        // LanceDB's HNSW happy (it requires fixed-width vectors).
        Field::new(
            ChunkColumn::EMBEDDING,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                embedding_dim,
            ),
            true,
        ),
        // Reserved for context tree. Nullable until then; populated
        // at retrieval time once context entries are added.
        Field::new(ChunkColumn::CONTEXT_ID, DataType::Utf8, true),
        // Per-row schema version mirror of the dataset metadata. Lets queries
        // tolerate mid-flight migrations without reading the manifest.
        Field::new(ChunkColumn::SCHEMA_VERSION, DataType::Int32, false),
    ]))
}

/// Default dataset path: `~/.markq/chunks.lance`.
///
/// Falls back to `./.markq/chunks.lance` when `dirs::home_dir()` is unset
/// (CI sandboxes, containers without `$HOME`).
pub fn default_dataset_path() -> PathBuf {
    let mut p = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    p.push(".markq");
    p.push("chunks.lance");
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_field_count_and_order_is_stable() {
        let s = chunk_arrow_schema(EMBEDDING_DIM_DEFAULT);
        let names: Vec<&str> = s.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec![
                "id",
                "collection",
                "uri",
                "path",
                "content_hash",
                "mtime",
                "chunk_index",
                "text",
                "tokens",
                "embedding",
                "context_id",
                "schema_version",
            ],
            "schema field order changed — this is a Lance migration, not a refactor"
        );
    }

    #[test]
    fn embedding_and_context_columns_are_nullable() {
        let s = chunk_arrow_schema(EMBEDDING_DIM_DEFAULT);
        let field = |n: &str| s.field_with_name(n).unwrap();
        assert!(field("embedding").is_nullable());
        assert!(field("context_id").is_nullable());
        assert!(field("tokens").is_nullable());
        assert!(!field("id").is_nullable());
        assert!(!field("path").is_nullable());
    }
}
