//! Re-exports of the CLI's internal modules so integration tests can drive
//! `markq index` / `markq search` without spawning the binary.

pub mod indexer;
pub mod search;
