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
use markq_inference::{default_n_gpu_layers, ensure_model, Embedder, KnownModel};

use crate::search::{apply_filters, SearchOptions};

pub struct ExplainTrace {
    pub bm25_hits: usize,
    pub bm25_ms: u128,
    pub embed_ms: u128,
    pub vector_hits: usize,
    pub vector_ms: u128,
    pub fuse_ms: u128,
    pub fused: Vec<FusedHit>,
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

    let k_pre = pre_fusion_k(k);

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
    let filtered = apply_filters(as_hits, opts);

    let explain_trace = if explain {
        Some(ExplainTrace {
            bm25_hits: bm25_hits.len(),
            bm25_ms,
            embed_ms,
            vector_hits: vec_hits.len(),
            vector_ms,
            fuse_ms,
            fused,
        })
    } else {
        None
    };

    Ok(QueryOutcome {
        hits: filtered,
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
    writeln!(w)?;
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
