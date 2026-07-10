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

    let order = order_by_relevance(&scores, top_k);
    let reranked: Vec<RerankedCandidate> = order
        .into_iter()
        .enumerate()
        .map(|(i, orig_idx)| {
            let rank = (i + 1) as u32;
            let score = scores[orig_idx];
            RerankedCandidate::new(candidates[orig_idx].clone(), score, rank)
        })
        .collect();

    if json {
        serde_json::to_writer_pretty(&mut *writer, &reranked)?;
        writeln!(writer)?;
    } else {
        write_table(writer, &reranked)?;
    }

    Ok(())
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

/// Collapse a candidate's text to a single-line, whitespace-normalized
/// preview truncated to `max_chars`, mirroring `search.rs`'s table preview.
fn preview_line(text: &str, max_chars: usize) -> String {
    let collapsed: String = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if collapsed.chars().count() <= max_chars {
        return collapsed;
    }
    let cut: String = collapsed.chars().take(max_chars).collect();
    format!("{cut}…")
}
