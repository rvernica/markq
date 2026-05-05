//! Dataset-level metadata keys (LanceDB `update_config` k/v) and the typed
//! struct `markq inspect` returns. Tracked in PHASE1_FOLLOWUPS.md item #4.
//!
//! Keys are versioned with the `markq.` prefix so downstream tools (`pylance`,
//! `duckdb lance_scan`, the bare `lance` CLI) can identify them as ours.

use serde::{Deserialize, Serialize};

/// Bumped when the chunk Arrow schema changes shape. Phase 1 ships v1.
pub const SCHEMA_VERSION: u32 = 1;

pub const KEY_SCHEMA_VERSION: &str = "markq.schema_version";

/// Lance dataset commit version (u64) — increments per write. Recorded for
/// debugging / time-travel; not used for compatibility decisions.
pub const KEY_LANCE_MANIFEST_VERSION: &str = "markq.lance_manifest_version";

/// Lance on-disk file format version (e.g. "2.1"). This is the migration-
/// relevant version: Lance bumps it when the storage format changes.
pub const KEY_LANCE_FILE_FORMAT_VERSION: &str = "markq.lance_file_format_version";

/// Version of the `lancedb` crate that created the dataset. Captured at
/// build time via `env!("CARGO_PKG_VERSION")` against our pinned dep.
pub const KEY_LANCEDB_CRATE_VERSION: &str = "markq.lancedb_crate_version";

/// Reserved for Phase 4 — embedder model id (e.g. `Qwen3-Embedding-0.6B`).
pub const KEY_EMBEDDER_MODEL: &str = "markq.embedder_model";

/// Reserved for Phase 4 — embedder output dimension as a decimal string.
pub const KEY_EMBEDDER_DIM: &str = "markq.embedder_dim";

/// Typed view of the metadata read back from a dataset. Optional fields are
/// the v1.5+ keys reserved but not yet populated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub schema_version: u32,
    pub lance_manifest_version: u64,
    pub lance_file_format_version: String,
    pub lancedb_crate_version: String,
    pub embedder_model: Option<String>,
    pub embedder_dim: Option<u32>,
}
