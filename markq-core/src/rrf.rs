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

pub fn fuse(lists: &[(&'static str, &[ChunkHit])], cfg: &FusionConfig) -> Vec<FusedHit> {
    use std::collections::HashMap;

    let mut by_id: HashMap<String, FusedHit> = HashMap::new();
    let mut order: Vec<String> = Vec::new();

    for (source, list) in lists {
        let weight = cfg.weights.get(source).copied().unwrap_or(0.0);
        for (idx, h) in list.iter().enumerate() {
            let rank = idx + 1;
            let rrf_value = weight / (cfg.k as f32 + rank as f32);
            let bonus = if rank <= cfg.top_rank_bonus.len() {
                cfg.top_rank_bonus[rank - 1]
            } else {
                0.0
            };
            let contribution = Contribution {
                source,
                rank,
                weight,
                rrf_value,
                bonus,
            };
            match by_id.get_mut(&h.id) {
                Some(existing) => {
                    existing.final_score += rrf_value + bonus;
                    existing.contributions.push(contribution);
                }
                None => {
                    order.push(h.id.clone());
                    by_id.insert(
                        h.id.clone(),
                        FusedHit {
                            hit: h.clone(),
                            final_score: rrf_value + bonus,
                            contributions: vec![contribution],
                        },
                    );
                }
            }
        }
    }

    let mut out: Vec<FusedHit> = order
        .into_iter()
        .map(|id| by_id.remove(&id).expect("seeded above"))
        .collect();

    out.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.hit.id.cmp(&b.hit.id))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(id: &str) -> ChunkHit {
        ChunkHit {
            id: id.to_string(),
            path: format!("{id}.md"),
            uri: format!("file:///{id}.md"),
            chunk_index: 0,
            text: String::new(),
            score: 0.0,
        }
    }

    #[test]
    fn single_list_preserves_order() {
        let lex = vec![hit("a"), hit("b"), hit("c")];
        let cfg = FusionConfig::default();
        let fused = fuse(&[("lex", &lex)], &cfg);
        let ids: Vec<&str> = fused.iter().map(|f| f.hit.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[test]
    fn two_lists_hand_computed() {
        // lex: [a, b, c]  vec: [b, a, d]
        // cfg defaults: k=60, w_lex=0.75, w_vec=0.60, bonus=[0.05, 0.02, 0.02]
        //
        // a: lex rank 1 -> 0.75/61 + 0.05  = 0.01229508 + 0.05
        //    vec rank 2 -> 0.60/62 + 0.02  = 0.00967742 + 0.02
        // b: lex rank 2 -> 0.75/62 + 0.02  = 0.01209677 + 0.02
        //    vec rank 1 -> 0.60/61 + 0.05  = 0.00983607 + 0.05
        // c: lex rank 3 -> 0.75/63 + 0.02  = 0.01190476 + 0.02
        // d: vec rank 3 -> 0.60/63 + 0.02  = 0.00952381 + 0.02
        let lex = vec![hit("a"), hit("b"), hit("c")];
        let vec_list = vec![hit("b"), hit("a"), hit("d")];
        let cfg = FusionConfig::default();
        let fused = fuse(&[("lex", &lex), ("vec", &vec_list)], &cfg);

        let ids: Vec<&str> = fused.iter().map(|f| f.hit.id.as_str()).collect();
        // a and b both appear in both lists; a is rank 1 in lex (heavier weight),
        // b is rank 1 in vec. With w_lex > w_vec, a should outscore b.
        assert_eq!(ids, vec!["a", "b", "c", "d"]);

        let a = &fused[0];
        let expected_a = 0.75 / 61.0 + 0.05 + 0.60 / 62.0 + 0.02;
        assert!(
            (a.final_score - expected_a).abs() < 1e-6,
            "a final_score {} vs expected {}",
            a.final_score,
            expected_a,
        );
        assert_eq!(a.contributions.len(), 2);
        assert_eq!(a.contributions[0].source, "lex");
        assert_eq!(a.contributions[0].rank, 1);
        assert_eq!(a.contributions[1].source, "vec");
        assert_eq!(a.contributions[1].rank, 2);
    }
}
