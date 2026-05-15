//! Singleton `LlamaBackend`. The C++ side maintains global ggml state and
//! refuses a second `llama_backend_init` (the upstream crate returns
//! `BackendAlreadyInitialized` on the second call); we initialize once per
//! process and hand out shared references.
//!
//! At init time we also route llama.cpp's stderr stream into `tracing` via
//! `llama-cpp-2::send_logs_to_tracing`. That covers PHASE1_FOLLOWUPS #5:
//! the cosmetic `embeddings required but…` per-decode line flows through
//! the `tracing` filter and disappears at the default `warn` level.

use anyhow::Context;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};
use once_cell::sync::OnceCell;
use tracing::debug;

static BACKEND: OnceCell<LlamaBackend> = OnceCell::new();

/// Obtain (initializing on first call) the process-wide `LlamaBackend`.
pub fn shared() -> anyhow::Result<&'static LlamaBackend> {
    BACKEND.get_or_try_init(|| {
        // Run *before* the backend init so the init banner itself flows
        // through `tracing`. `send_logs_to_tracing` is one-shot per process;
        // re-calling is documented as a no-op.
        send_logs_to_tracing(LogOptions::default());
        let backend = LlamaBackend::init().context("LlamaBackend::init")?;
        debug!("LlamaBackend initialized");
        Ok::<_, anyhow::Error>(backend)
    })
}
