//! Weighted Reciprocal Rank Fusion (RRF).
//!
//! Combines multiple ranked retrieval lists (BM25, vector, ...) into one
//! ranking. Per-list weights and a small bonus for the top three ranks are
//! configurable; defaults match the values pinned in `plan.md`.

use std::collections::BTreeMap;

use crate::ChunkHit;

/// Per-source contribution to a fused score, captured for the `--explain`
/// trace.
#[derive(Debug, Clone, PartialEq)]
pub struct Contribution {
    pub source: &'static str,
    /// 1-based rank within the source list.
    pub rank: usize,
    pub weight: f32,
    /// `weight / (k + rank as f32)`.
    pub rrf_value: f32,
    /// Top-rank bonus contribution; 0 if `rank > top_rank_bonus.len()`.
    pub bonus: f32,
}

#[derive(Debug, Clone)]
pub struct FusedHit {
    pub hit: ChunkHit,
    pub final_score: f32,
    pub contributions: Vec<Contribution>,
}

#[derive(Debug, Clone)]
pub struct FusionConfig {
    pub k: usize,
    pub weights: BTreeMap<&'static str, f32>,
    pub top_rank_bonus: [f32; 3],
}

impl Default for FusionConfig {
    fn default() -> Self {
        let mut weights = BTreeMap::new();
        weights.insert("lex", 0.75);
        weights.insert("vec", 0.60);
        Self {
            k: 60,
            weights,
            top_rank_bonus: [0.05, 0.02, 0.02],
        }
    }
}

pub fn fuse(_lists: &[(&'static str, &[ChunkHit])], _cfg: &FusionConfig) -> Vec<FusedHit> {
    Vec::new()
}
