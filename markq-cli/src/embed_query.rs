//! Embed a single query string and print the vector. Lets external tools
//! (DuckDB's `lance_vector_search`, Python via pylance, jq pipelines)
//! run their own vector search against the markq dataset without
//! needing to load a GGUF themselves.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use markq_core::Index;
use markq_index_lance::LanceIndex;
use markq_inference::{ensure_model, Embedder, KnownModel};

/// Embed `query` with the canonical embedder and write the vector as a
/// single-line JSON array to `out`. If `dataset_path` points at an
/// existing dataset that already records an `embedder_model`, the model
/// IDs must match — otherwise the printed vector would not be cosine-
/// compatible with the dataset's stored vectors.
pub async fn run_embed_query<W: Write>(
    dataset_path: &Path,
    query: &str,
    out: &mut W,
) -> Result<()> {
    let model = KnownModel::Qwen3Embedding06B;

    if dataset_path.exists() {
        let idx = LanceIndex::open(dataset_path)
            .await
            .context("open dataset")?;
        let md = idx.metadata().await.context("read dataset metadata")?;
        if let Some(existing) = md.embedder_model.as_deref() {
            if existing != model.id() {
                anyhow::bail!(
                    "dataset was built with embedder {existing}, but this build only knows {}",
                    model.id()
                );
            }
        }
    }

    let model_path = ensure_model(model).await.context("locate embedder model")?;
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    let n_gpu_layers: u32 = 999;
    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    let n_gpu_layers: u32 = 0;
    let embedder = Embedder::load(&model_path, n_gpu_layers).context("load embedder")?;
    let vec = embedder
        .embed(query.to_string())
        .await
        .context("embed query")?;

    write!(out, "[")?;
    for (i, v) in vec.iter().enumerate() {
        if i > 0 {
            write!(out, ",")?;
        }
        write!(out, "{v}")?;
    }
    writeln!(out, "]")?;
    Ok(())
}
