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

    for (source, list) in lists {
        let weight = cfg.weights.get(source).copied().unwrap_or(0.0);
        for (idx, h) in list.iter().enumerate() {
            let rank = idx + 1;
            let rrf_value = weight / (cfg.k as f32 + rank as f32);
            // Bonus is applied per source contribution, so a hit ranked
            // within `top_rank_bonus.len()` in multiple lists accumulates
            // the bonus once per list (e.g. rank 1 in both lex and vec
            // earns the rank-1 bonus twice).
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

    let mut out: Vec<FusedHit> = by_id.into_values().collect();
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

    #[test]
    fn bonus_applies_to_top_three_only() {
        let lex: Vec<ChunkHit> = (0..5).map(|i| hit(&format!("d{i}"))).collect();
        let cfg = FusionConfig::default();
        let fused = fuse(&[("lex", &lex)], &cfg);

        // d0..d2 (ranks 1..3) get a bonus; d3, d4 do not.
        assert!(fused[0].contributions[0].bonus > 0.0);
        assert!(fused[1].contributions[0].bonus > 0.0);
        assert!(fused[2].contributions[0].bonus > 0.0);
        assert_eq!(fused[3].contributions[0].bonus, 0.0);
        assert_eq!(fused[4].contributions[0].bonus, 0.0);
    }

    #[test]
    fn doc_in_only_one_list_gets_one_contribution() {
        let lex = vec![hit("solo")];
        let vec_list: Vec<ChunkHit> = Vec::new();
        let cfg = FusionConfig::default();
        let fused = fuse(&[("lex", &lex), ("vec", &vec_list)], &cfg);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].contributions.len(), 1);
        assert_eq!(fused[0].contributions[0].source, "lex");
    }

    #[test]
    fn rank_one_in_both_beats_rank_one_in_only_one() {
        // `both` is rank 1 in both lists. `only_lex` is rank 1 in lex but
        // absent from vec. With positive weights for both sources, `both`
        // must outscore `only_lex`.
        let lex = vec![hit("both"), hit("only_lex")];
        let vec_list = vec![hit("both"), hit("filler")];
        let cfg = FusionConfig::default();
        let fused = fuse(&[("lex", &lex), ("vec", &vec_list)], &cfg);

        let score = |id: &str| {
            fused
                .iter()
                .find(|f| f.hit.id == id)
                .map(|f| f.final_score)
                .expect("present")
        };
        assert!(
            score("both") > score("only_lex"),
            "both={} vs only_lex={}",
            score("both"),
            score("only_lex"),
        );
    }

    #[test]
    fn rank_one_bonus_accumulates_per_source() {
        // A doc at rank 1 in both lists earns the rank-1 bonus twice
        // (once per source contribution). With defaults (0.05, 0.05) the
        // bonus total on the fused hit must be 0.10.
        let lex = vec![hit("x")];
        let vec_list = vec![hit("x")];
        let cfg = FusionConfig::default();
        let fused = fuse(&[("lex", &lex), ("vec", &vec_list)], &cfg);
        assert_eq!(fused.len(), 1);
        let total_bonus: f32 = fused[0].contributions.iter().map(|c| c.bonus).sum();
        assert!(
            (total_bonus - 0.10).abs() < 1e-6,
            "expected bonus 0.10, got {total_bonus}",
        );
    }

    #[test]
    fn equal_final_score_breaks_ties_by_id_ascending() {
        // Two docs with the same final_score must sort by id ascending,
        // regardless of HashMap iteration order. Build a config where the
        // two sources are weighted identically and feed each doc at the
        // same rank in its own source — final_score collides exactly.
        let mut weights = std::collections::BTreeMap::new();
        weights.insert("lex", 0.5);
        weights.insert("vec", 0.5);
        let cfg = FusionConfig {
            k: 60,
            weights,
            top_rank_bonus: [0.0, 0.0, 0.0],
        };
        let lex = vec![hit("z_late")];
        let vec_list = vec![hit("a_early")];
        let fused = fuse(&[("lex", &lex), ("vec", &vec_list)], &cfg);

        assert_eq!(fused.len(), 2);
        assert!(
            (fused[0].final_score - fused[1].final_score).abs() < 1e-6,
            "scores should tie: {} vs {}",
            fused[0].final_score,
            fused[1].final_score,
        );
        assert_eq!(fused[0].hit.id, "a_early");
        assert_eq!(fused[1].hit.id, "z_late");
    }
}
