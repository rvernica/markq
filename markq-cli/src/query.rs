//! Hybrid retrieval: BM25 + vector embedded query + weighted RRF fusion.
//!
//! Reuses `search::{Format, SearchOptions, apply_filters, write_results}`
//! for output, so `--json` / `--files` / `--all` / `--top-k` (`-n`) / `--min-score`
//! behave identically to `markq search` and `markq vsearch`.

use std::io::Write;
use std::time::Instant;

use anyhow::{Context, Result};
use markq_core::{fuse, ChunkHit, FusedHit, FusionConfig, Index};
use markq_index_lance::LanceIndex;
use markq_inference::{default_n_gpu_layers, ensure_model, Embedder, KnownModel, Reranker};
use tracing::warn;

use crate::search::{apply_filters, SearchOptions};

/// Cap on how many fused hits the cross-encoder reranks. Rerank is a
/// precision SECOND stage over a small candidate set (~20-50, interactive
/// speed), not a primary retriever, so an unbounded `--all` or large
/// fan-out must not turn it into an O(pool) cost. When the fused pool
/// exceeds this, only the top `RERANK_FANIN` fused hits (already in
/// fusion order) are reranked; the truncation is surfaced via `warn!`
/// rather than dropped silently. This also bounds the RESULT COUNT under
/// `--rerank`: a larger `--top-k` (`-n`) or `--all` cannot yield more than
/// `RERANK_FANIN` results, unlike the non-rerank path.
const RERANK_FANIN: usize = 64;

pub struct ExplainTrace {
    pub bm25_hits: usize,
    pub bm25_ms: u128,
    pub embed_ms: u128,
    pub vector_hits: usize,
    pub vector_ms: u128,
    pub fuse_ms: u128,
    pub fused: Vec<FusedHit>,
    /// Time spent scoring candidates in the rerank stage, and how many
    /// were scored. `None` when `--rerank` was not requested.
    pub rerank_ms: Option<u128>,
    pub rerank_candidates: Option<usize>,
}

pub struct QueryOutcome {
    pub hits: Vec<ChunkHit>,
    pub explain: Option<ExplainTrace>,
}

/// Pre-fusion fetch depth. Over-fetches each retrieval list so the fused
/// top-k has real candidates from both sides when the lists disagree near
/// the head. Doubles the requested `k` with a floor of 20, so small queries
/// still get a meaningful margin and large queries keep one.
fn pre_fusion_k(k: usize) -> usize {
    k.saturating_mul(2).max(20)
}

pub async fn run_query(
    idx: &LanceIndex,
    query: &str,
    k: usize,
    opts: &SearchOptions,
    explain: bool,
    rerank: bool,
) -> Result<QueryOutcome> {
    let md = idx.metadata().await.context("read dataset metadata")?;
    if md.embedder_model.is_none() || md.embedder_dim.is_none() {
        anyhow::bail!("no embeddings in this dataset; run `markq embed` first to populate them");
    }
    let model = KnownModel::Qwen3Embedding06B;
    let existing = md
        .embedder_model
        .as_deref()
        .expect("guarded by is_none() check above");
    if existing != model.id() {
        anyhow::bail!(
            "dataset was built with embedder {existing}, but this build only knows {}",
            model.id()
        );
    }

    let model_path = ensure_model(model).await.context("locate embedder model")?;
    let embedder = Embedder::load(&model_path, default_n_gpu_layers()).context("load embedder")?;

    // When reranking, the cross-encoder consumes at most `RERANK_FANIN` fused
    // candidates, so bound the pre-fusion fetch depth to match. Otherwise
    // `--rerank --all` (or a large `--top-k` (`-n`)) would make BM25 + vector retrieval and
    // fusion do O(dataset) work only to discard everything past the fan-in cap.
    // `pre_fusion_k` still doubles this, leaving ample headroom to fill the
    // fan-in window. The non-rerank path is unchanged.
    let effective_k = if rerank { k.min(RERANK_FANIN) } else { k };
    let k_pre = pre_fusion_k(effective_k);

    let bm25_fut = async {
        let t = Instant::now();
        let r = idx.bm25(query, k_pre, None).await;
        (r, t.elapsed().as_millis())
    };
    let embed_fut = async {
        let t = Instant::now();
        let r = embedder.embed(query.to_string()).await;
        (r, t.elapsed().as_millis())
    };
    let ((bm25_res, bm25_ms), (embed_res, embed_ms)) = tokio::join!(bm25_fut, embed_fut);
    let bm25_hits = bm25_res.context("bm25 search")?;
    let q_vec = embed_res.context("embed query")?;
    // `embedder` is only needed to produce `q_vec`; drop its resident
    // `LlamaContext` now so the `--rerank` branch doesn't hold both the
    // embedder and reranker models in memory at once.
    drop(embedder);

    let vec_t = Instant::now();
    let vec_hits = idx
        .vector(&q_vec, k_pre, None)
        .await
        .context("vector search")?;
    let vector_ms = vec_t.elapsed().as_millis();

    let fuse_t = Instant::now();
    let cfg = FusionConfig::default();
    let fused = fuse(&[("lex", &bm25_hits), ("vec", &vec_hits)], &cfg);
    let fuse_ms = fuse_t.elapsed().as_millis();

    let as_hits: Vec<ChunkHit> = fused
        .iter()
        .map(|f| ChunkHit {
            score: f.final_score,
            ..f.hit.clone()
        })
        .collect();
    let mut rerank_ms = None;
    let mut rerank_candidates = None;
    let hits = if rerank {
        // Reranking runs strictly AFTER fusion, but only over the top
        // `RERANK_FANIN` fused hits (`as_hits` is already in fusion
        // order) -- rerank is a small-set precision stage, not a
        // primary retriever, so the final top-k reflects the
        // cross-encoder's judgment over a bounded fan-in.
        if as_hits.len() > RERANK_FANIN {
            warn!(
                pool = as_hits.len(),
                cap = RERANK_FANIN,
                "rerank input truncated to RERANK_FANIN fused hits"
            );
        }
        let fanin = as_hits.len().min(RERANK_FANIN);
        let rerank_model_path = ensure_model(KnownModel::Qwen3Reranker06B)
            .await
            .context("locate reranker model")?;
        let reranker =
            Reranker::load(&rerank_model_path, default_n_gpu_layers()).context("load reranker")?;
        let rerank_t = Instant::now();
        let mut scores = Vec::with_capacity(fanin);
        for h in &as_hits[..fanin] {
            scores.push(
                reranker
                    .score(query, &h.text, None)
                    .await
                    .context("rerank score")?,
            );
        }
        rerank_ms = Some(rerank_t.elapsed().as_millis());
        rerank_candidates = Some(fanin);
        let order = markq_inference::order_by_relevance(&scores, None);
        let reordered: Vec<ChunkHit> = order
            .iter()
            .map(|&i| ChunkHit {
                score: scores[i],
                ..as_hits[i].clone()
            })
            .collect();
        apply_filters(reordered, opts)
    } else {
        apply_filters(as_hits, opts)
    };

    let explain_trace = if explain {
        Some(ExplainTrace {
            bm25_hits: bm25_hits.len(),
            bm25_ms,
            embed_ms,
            vector_hits: vec_hits.len(),
            vector_ms,
            fuse_ms,
            fused,
            rerank_ms,
            rerank_candidates,
        })
    } else {
        None
    };

    Ok(QueryOutcome {
        hits,
        explain: explain_trace,
    })
}

pub fn write_explain<W: Write>(w: &mut W, trace: &ExplainTrace, shown: &[ChunkHit]) -> Result<()> {
    writeln!(w, "bm25:   {} hits in {}ms", trace.bm25_hits, trace.bm25_ms)?;
    writeln!(w, "embed:  query in {}ms", trace.embed_ms)?;
    writeln!(
        w,
        "vector: {} hits in {}ms",
        trace.vector_hits, trace.vector_ms
    )?;
    writeln!(
        w,
        "fuse:   {} unique docs in {}ms",
        trace.fused.len(),
        trace.fuse_ms
    )?;
    if let (Some(ms), Some(n)) = (trace.rerank_ms, trace.rerank_candidates) {
        writeln!(w, "rerank: {n} candidates in {ms}ms")?;
    }
    writeln!(w)?;
    if trace.rerank_ms.is_some() {
        writeln!(
            w,
            "note: rows below are ordered by cross-encoder rerank relevance, \
             not by the `final` column; `final` still shows the pre-rerank RRF fusion score."
        )?;
        writeln!(w)?;
    }
    writeln!(
        w,
        "{:<4}  {:<12}  {:>8}  {:>15}  {:>15}  {:>6}",
        "rank", "id", "final", "lex(rank,w)", "vec(rank,w)", "bonus"
    )?;

    let mut by_id = std::collections::HashMap::new();
    for f in &trace.fused {
        by_id.insert(f.hit.id.clone(), f);
    }

    for (i, h) in shown.iter().enumerate() {
        let Some(f) = by_id.get(&h.id) else { continue };
        let cell = |source: &str| -> String {
            f.contributions
                .iter()
                .find(|c| c.source == source)
                .map(|c| format!("({:>2}, {:.2})", c.rank, c.weight))
                .unwrap_or_else(|| "(  - ,   - )".to_string())
        };
        let bonus_total: f32 = f.contributions.iter().map(|c| c.bonus).sum();
        let id_short: String = h.id.chars().take(12).collect();
        writeln!(
            w,
            "{:<4}  {:<12}  {:>8.4}  {:>15}  {:>15}  {:>6.2}",
            i + 1,
            id_short,
            f.final_score,
            cell("lex"),
            cell("vec"),
            bonus_total,
        )?;
    }
    Ok(())
}
