//! `markq rerank`: read a `Vec<Candidate>` from stdin, score each (query,
//! document) pair with the cross-encoder reranker, reorder best-first, and
//! emit either a human ranked list or `RerankedCandidate` JSON.
//!
//! Unlike `search`/`vsearch`/`query`, this command never opens the Lance
//! dataset — it operates purely on stdin candidates, so it composes with any
//! upstream retriever that can emit the `Candidate` wire shape.

use std::io::{Read, Write};

use anyhow::{Context, Result};
use markq_inference::{
    default_n_gpu_layers, ensure_model, order_by_relevance, KnownModel, Reranker,
};

use crate::candidate::{Candidate, RerankedCandidate};
use crate::search::preview_line;

/// Read `Vec<Candidate>` JSON from `reader`, score each against `query` with
/// the cross-encoder reranker, and write the reordered result to `writer`.
///
/// Candidates are scored sequentially in input order (the reranker owner
/// thread serializes requests anyway, so this stays deterministic without
/// giving up any concurrency). `top_k` truncates the sorted output; `None`
/// keeps every candidate.
pub async fn run_rerank<R: Read, W: Write>(
    reader: R,
    writer: &mut W,
    query: &str,
    top_k: Option<usize>,
    instruction: Option<&str>,
    json: bool,
) -> Result<()> {
    let candidates: Vec<Candidate> =
        serde_json::from_reader(reader).context("parse candidates JSON from stdin")?;

    let model_path = ensure_model(KnownModel::Qwen3Reranker06B)
        .await
        .context("locate reranker model")?;
    let reranker = Reranker::load(&model_path, default_n_gpu_layers()).context("load reranker")?;

    let mut scores = Vec::with_capacity(candidates.len());
    for candidate in &candidates {
        let score = reranker
            .score(query, &candidate.text, instruction)
            .await
            .with_context(|| format!("score candidate {}", candidate.id))?;
        scores.push(score);
    }

    let reranked = assemble_ranked(candidates, &scores, top_k);

    if json {
        serde_json::to_writer_pretty(&mut *writer, &reranked)?;
        writeln!(writer)?;
    } else {
        write_table(writer, &reranked)?;
    }

    Ok(())
}

/// Model-independent core of `run_rerank`: given `candidates` and their
/// already-computed cross-encoder `scores` (index-aligned with
/// `candidates`), sort best-first via [`order_by_relevance`], truncate to
/// `top_k`, and assign 1-based ranks. Split out from `run_rerank` so this
/// logic is covered by plain unit tests that don't need a loaded model.
fn assemble_ranked(
    candidates: Vec<Candidate>,
    scores: &[f32],
    top_k: Option<usize>,
) -> Vec<RerankedCandidate> {
    let order = order_by_relevance(scores, top_k);
    // `order_by_relevance` only returns indices, so pull each candidate out
    // of `candidates` by index; `swap_remove`-free indexing needs `Clone`
    // since a candidate's position in `order` isn't guaranteed to be
    // consumed in original order.
    order
        .into_iter()
        .enumerate()
        .map(|(i, orig_idx)| {
            let rank = (i + 1) as u32;
            let score = scores[orig_idx];
            RerankedCandidate::new(candidates[orig_idx].clone(), score, rank)
        })
        .collect()
}

fn write_table<W: Write>(w: &mut W, reranked: &[RerankedCandidate]) -> Result<()> {
    if reranked.is_empty() {
        writeln!(w, "(no results)")?;
        return Ok(());
    }
    for r in reranked {
        writeln!(
            w,
            "{rank:>3}. {score:>7.3}  {id}",
            rank = r.rank,
            score = r.rerank_score,
            id = r.id,
        )?;
        writeln!(w, "     {}", preview_line(&r.text, 100))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Map, Value};

    fn candidate(id: &str, text: &str, extra: Map<String, Value>) -> Candidate {
        Candidate {
            id: id.to_string(),
            text: text.to_string(),
            extra,
        }
    }

    #[test]
    fn assigns_1_based_rank_in_score_order() {
        let candidates = vec![
            candidate("a", "alpha", Map::new()),
            candidate("b", "beta", Map::new()),
            candidate("c", "gamma", Map::new()),
        ];
        // b scores highest, then c, then a.
        let scores = [0.1, 0.9, 0.5];

        let reranked = assemble_ranked(candidates, &scores, None);

        assert_eq!(reranked.len(), 3);
        assert_eq!(reranked[0].id, "b");
        assert_eq!(reranked[0].rank, 1);
        assert_eq!(reranked[0].rerank_score, 0.9);
        assert_eq!(reranked[1].id, "c");
        assert_eq!(reranked[1].rank, 2);
        assert_eq!(reranked[2].id, "a");
        assert_eq!(reranked[2].rank, 3);
    }

    #[test]
    fn top_k_truncates_the_sorted_output() {
        let candidates = vec![
            candidate("a", "alpha", Map::new()),
            candidate("b", "beta", Map::new()),
            candidate("c", "gamma", Map::new()),
        ];
        let scores = [0.1, 0.9, 0.5];

        let reranked = assemble_ranked(candidates, &scores, Some(2));

        assert_eq!(reranked.len(), 2);
        assert_eq!(reranked[0].id, "b");
        assert_eq!(reranked[1].id, "c");
    }

    #[test]
    fn passthrough_fields_survive_and_stale_rank_keys_are_deduped() {
        let mut extra = Map::new();
        extra.insert("score".to_string(), json!(0.42));
        extra.insert("collection".to_string(), json!("docs"));
        // Simulate piping a previous `markq rerank --json` output back in:
        // the input candidate's `extra` already has stale rank fields.
        extra.insert("rank".to_string(), json!(7));
        extra.insert("rerank_score".to_string(), json!(0.123));

        let candidates = vec![candidate("a", "alpha", extra)];
        let scores = [0.75];

        let reranked = assemble_ranked(candidates, &scores, None);

        assert_eq!(reranked.len(), 1);
        let r = &reranked[0];
        assert_eq!(r.rank, 1);
        assert_eq!(r.rerank_score, 0.75);
        assert_eq!(r.extra.get("score"), Some(&json!(0.42)));
        assert_eq!(r.extra.get("collection"), Some(&json!("docs")));
        // The stale passthrough keys must not survive in `extra` — otherwise
        // serializing `RerankedCandidate` would emit `rank`/`rerank_score`
        // twice (once from the typed field, once from the flattened map).
        assert!(!r.extra.contains_key("rank"));
        assert!(!r.extra.contains_key("rerank_score"));

        let serialized = serde_json::to_value(r).unwrap();
        assert_eq!(serialized["rank"], json!(1));
        assert_eq!(serialized["rerank_score"], json!(0.75));
        assert_eq!(serialized["score"], json!(0.42));

        // Round-trip through a string to make sure there's no duplicate key
        // in the actual JSON text (serde_json::Value would silently keep
        // only the last occurrence, masking the bug at the Value level).
        let text = serde_json::to_string(r).unwrap();
        assert_eq!(
            text.matches("\"rank\"").count(),
            1,
            "duplicate \"rank\" key in serialized JSON: {text}"
        );
        assert_eq!(
            text.matches("\"rerank_score\"").count(),
            1,
            "duplicate \"rerank_score\" key in serialized JSON: {text}"
        );
    }

    #[test]
    fn json_output_is_a_well_formed_array() {
        let candidates = vec![candidate("a", "alpha", Map::new())];
        let scores = [0.5];
        let reranked = assemble_ranked(candidates, &scores, None);

        let mut buf = Vec::new();
        serde_json::to_writer_pretty(&mut buf, &reranked).unwrap();
        let v: Value = serde_json::from_slice(&buf).unwrap();
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["id"], "a");
        assert_eq!(arr[0]["rank"], 1);
    }

    #[test]
    fn table_output_renders_rank_score_and_preview() {
        let candidates = vec![candidate("a", "alpha text here", Map::new())];
        let scores = [0.5];
        let reranked = assemble_ranked(candidates, &scores, None);

        let mut buf = Vec::new();
        write_table(&mut buf, &reranked).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("1."));
        assert!(out.contains("0.500"));
        assert!(out.contains("alpha text here"));
    }

    #[test]
    fn table_output_reports_no_results_when_empty() {
        let mut buf = Vec::new();
        write_table(&mut buf, &[]).unwrap();
        assert_eq!(String::from_utf8(buf).unwrap(), "(no results)\n");
    }
}
