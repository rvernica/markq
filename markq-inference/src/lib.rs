//! Inference layer (embedder, reranker, optional generator). Stub for Phase 1.
//!
//! Phase 4: embedder thread (`llama-cpp-2`) + bounded `crossbeam-channel`,
//! 32–64-chunk batches per `llama_decode`. See plan.md § "Inference
//! architecture" — `LlamaContext` is `!Send`, so single-owner thread per
//! model is the only correct shape.
