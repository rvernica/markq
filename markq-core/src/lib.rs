//! Core types for markq: chunk schema, `Index` trait, registry, errors.
//!
//! Everything here is backend-agnostic. The concrete index implementation
//! lives in `markq-index-lance`.

#[cfg(any(test, feature = "test-harness"))]
pub mod contract;

pub mod error;
pub mod index;
pub mod metadata;
pub mod registry;
pub mod schema;

pub use error::{Error, Result};
pub use index::{ChunkHit, Index};
pub use metadata::{
    DatasetMetadata, KEY_EMBEDDER_DIM, KEY_EMBEDDER_MODEL, KEY_LANCEDB_CRATE_VERSION,
    KEY_LANCE_FILE_FORMAT_VERSION, KEY_LANCE_MANIFEST_VERSION, KEY_SCHEMA_VERSION, SCHEMA_VERSION,
};
pub use registry::{CollectionEntry, Registry};
pub use schema::{
    chunk_arrow_schema, default_dataset_path, markq_home, ChunkColumn, EMBEDDING_DIM_DEFAULT,
};
