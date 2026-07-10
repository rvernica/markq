//! Inference layer: embedder today, with a reranker and optional HyDE
//! generator to follow. Each model runs on its own owner thread fed by a
//! bounded crossbeam channel; `LlamaContext` is `!Send` so this is the only
//! correct shape.

pub mod backend;
pub mod embedder;
pub mod model_cache;
pub mod reranker;

pub use embedder::Embedder;
pub use model_cache::{ensure_model, models_dir, sha256_hex, KnownModel};
pub use reranker::Reranker;

/// Default `n_gpu_layers` for `Embedder::load`. Returns 999 (offload every
/// layer) when the crate is built with the `vulkan` or `cuda` feature, 0
/// otherwise. Centralizes the GPU-offload policy so every call site picks
/// up a future change (e.g. honoring a low-VRAM env override) for free.
pub const fn default_n_gpu_layers() -> u32 {
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    {
        999
    }
    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    {
        0
    }
}
