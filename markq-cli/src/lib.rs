//! Re-exports of the CLI's internal modules so integration tests can drive
//! `markq index` / `markq search` / `markq embed` / `markq vsearch` without
//! spawning the binary.

pub mod embedder_cmd;
pub mod indexer;
pub mod search;
pub mod vsearch;
