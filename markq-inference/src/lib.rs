//! Inference layer: embedder (Phase 4), reranker (Phase 6), optional HyDE
//! generator (Phase 13). Each model runs on its own owner thread fed by a
//! bounded crossbeam channel; `LlamaContext` is `!Send` so this is the only
//! correct shape (see plan.md § "Inference architecture").

pub mod backend;
pub mod embedder;
pub mod model_cache;

pub use embedder::Embedder;
pub use model_cache::{ensure_model, models_dir, sha256_hex, KnownModel};
