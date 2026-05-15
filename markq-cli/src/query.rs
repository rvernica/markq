//! Hybrid retrieval: BM25 + vector embedded query + weighted RRF fusion.
//!
//! Reuses `search::{Format, SearchOptions, apply_filters, write_results}`
//! for output, so `--json` / `--files` / `--all` / `-n` / `--min-score`
//! behave identically to `markq search` and `markq vsearch`.

use std::io::Write;
use std::time::Instant;

use anyhow::{Context, Result};
use markq_core::{fuse, ChunkHit, FusedHit, FusionConfig, Index};
use markq_index_lance::LanceIndex;
use markq_inference::{ensure_model, Embedder, KnownModel};

use crate::search::{apply_filters, SearchOptions};

/// Per-stage timing and post-fusion trace, populated only when the CLI
/// passed `--explain`.
pub struct ExplainTrace {
    pub bm25_hits: usize,
    pub bm25_ms: u128,
    pub embed_ms: u128,
    pub vector_hits: usize,
    pub vector_ms: u128,
    pub fuse_ms: u128,
    /// Full fused list pre-truncation, for the trace table.
    pub fused: Vec<FusedHit>,
}

pub struct QueryOutcome {
    pub hits: Vec<ChunkHit>,
    pub explain: Option<ExplainTrace>,
}

pub async fn run_query(
    _idx: &LanceIndex,
    _query: &str,
    _k: usize,
    _opts: &SearchOptions,
    _explain: bool,
) -> Result<QueryOutcome> {
    anyhow::bail!("not yet implemented");
}

pub fn write_explain<W: Write>(_w: &mut W, _trace: &ExplainTrace, _shown: &[ChunkHit]) -> Result<()> {
    Ok(())
}

// Bring the inference crate in only to keep the imports honest until
// `run_query` lands; the warning will go away in Task 7.
#[allow(dead_code)]
fn _force_use(_: KnownModel, _: Embedder) {}
#[allow(dead_code)]
async fn _force_ensure(m: KnownModel) -> Result<()> {
    let _ = ensure_model(m).await.context("locate embedder model")?;
    Ok(())
}
