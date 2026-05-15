//! `markq vsearch`: embed the query, run vector KNN on Lance.
//!
//! Reuses the `search` module's formatters (`Format`, `SearchOptions`,
//! `apply_filters`, `write_results`) so output flags work identically to
//! `markq search`. The only difference is the retrieval algorithm.

use anyhow::{Context, Result};
use markq_core::Index;
use markq_index_lance::LanceIndex;
use markq_inference::{ensure_model, Embedder, KnownModel};

use crate::search::{apply_filters, SearchOptions};
use markq_core::ChunkHit;

/// Embed `query` and run vector KNN. `k` is the post-fetch cap; the caller
/// derives it from `--all` / `-n` / row count just like `markq search`.
///
/// Returns a `Vec<ChunkHit>` with scores in `[-1, 1]` where higher is more
/// similar (cosine similarity = `1 - cosine_distance`).
pub async fn run_vsearch(
    idx: &LanceIndex,
    query: &str,
    k: usize,
    opts: &SearchOptions,
) -> Result<Vec<ChunkHit>> {
    let md = idx.metadata().await.context("read dataset metadata")?;
    if md.embedder_model.is_none() || md.embedder_dim.is_none() {
        anyhow::bail!("no embeddings in this dataset; run `markq embed` first to populate them");
    }

    let model = KnownModel::Qwen3Embedding06B;
    let existing = md.embedder_model.as_deref().expect("guarded by is_none() check above");
    if existing != model.id() {
        anyhow::bail!(
            "dataset was built with embedder {existing}, but this build only knows {}",
            model.id()
        );
    }

    let model_path = ensure_model(model).await.context("locate embedder model")?;
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    let n_gpu_layers: u32 = 999;
    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    let n_gpu_layers: u32 = 0;
    let embedder = Embedder::load(&model_path, n_gpu_layers).context("load embedder")?;

    let q_vec = embedder
        .embed(query.to_string())
        .await
        .context("embed query")?;

    let raw = idx.vector(&q_vec, k, None).await.context("vector search")?;
    Ok(apply_filters(raw, opts))
}
