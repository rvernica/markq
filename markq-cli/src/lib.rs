//! Re-exports of the CLI's internal modules so integration tests can drive
//! `markq index` / `markq search` / `markq embed` / `markq vsearch` without
//! spawning the binary.

pub mod embed_query;
pub mod embedder_cmd;
pub mod indexer;
pub mod query;
pub mod search;
pub mod vsearch;
