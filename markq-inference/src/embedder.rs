//! Embedder: owner thread holding `LlamaModel + LlamaContext`, fed by a
//! bounded crossbeam channel. `LlamaContext` is `!Send`, so the only correct
//! shape is single-owner thread + message passing. A future reranker and
//! HyDE generator copy this exact shape.
//!
//! Batching: this first cut decodes one text per `llama_decode`. The plan
//! aspires to 32–64 chunks per decode for throughput; the channel + owner-
//! thread structure already supports it (a future change just needs to
//! accumulate requests up to `n_batch` tokens before calling decode). The
//! bounded channel applies back-pressure regardless of batch size.

use std::num::NonZeroU32;
use std::path::Path;
use std::thread::{self, JoinHandle};

use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{bounded, Sender};
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use tokio::sync::oneshot;
use tracing::{debug, error, warn};

use crate::backend;

/// Bounded queue capacity. Producer (CLI rayon pool) blocks once this many
/// requests are in flight; the owner thread drains them as fast as it can.
const CHANNEL_CAPACITY: usize = 256;

/// llama.cpp context size. Qwen3-Embedding-0.6B's training context is 32K but
/// markq chunks target ~900 tokens; 2048 is comfortable with headroom for
/// the BOS / instruction tokens.
const DEFAULT_N_CTX: u32 = 2048;

/// llama.cpp logical batch size. Sized large enough that even a worst-case
/// 900-token chunk fits in one decode.
const DEFAULT_N_BATCH: u32 = 2048;

struct Request {
    text: String,
    reply: oneshot::Sender<Result<Vec<f32>>>,
}

/// Handle to a running embedder. Drop to shut down the owner thread (the
/// channel sender is dropped, owner observes `Disconnected`, exits cleanly).
pub struct Embedder {
    tx: Sender<Request>,
    /// Owned model dim, surfaced via `dim()`. `LlamaModel::n_embd` returns
    /// `c_int`; we narrow to `u32` since negative dims are not a real thing.
    dim: u32,
    /// Joined on `Drop`; held in `Option` so `drop` can take it.
    handle: Option<JoinHandle<()>>,
}

impl Embedder {
    /// Load a GGUF and spawn the owner thread.
    ///
    /// `n_gpu_layers = 999` pushes every layer to the GPU when the underlying
    /// crate was built with the `vulkan` / `cuda` feature; on a CPU-only
    /// build the value is silently ignored.
    pub fn load(path: &Path, n_gpu_layers: u32) -> Result<Embedder> {
        let backend = backend::shared()?;
        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(backend, path, &model_params)
            .with_context(|| format!("load_from_file {}", path.display()))?;
        let dim =
            u32::try_from(model.n_embd()).map_err(|_| anyhow!("model reports negative n_embd"))?;

        let (tx, rx) = bounded::<Request>(CHANNEL_CAPACITY);
        let handle = thread::Builder::new()
            .name("markq-embedder".to_string())
            .spawn(move || run_owner(backend, model, rx))
            .context("spawn embedder thread")?;

        Ok(Embedder {
            tx,
            dim,
            handle: Some(handle),
        })
    }

    /// Embedding dimension, recorded in dataset metadata to detect mismatch
    /// on later opens.
    pub fn dim(&self) -> u32 {
        self.dim
    }

    /// Submit one text for embedding. Awaits the owner thread's reply.
    ///
    /// Blocks (via the bounded channel) when the queue is full; this is the
    /// back-pressure that keeps the producer pool from out-running inference.
    pub async fn embed(&self, text: String) -> Result<Vec<f32>> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Request { text, reply })
            .map_err(|_| anyhow!("embedder thread is gone"))?;
        rx.await.map_err(|_| anyhow!("embedder dropped reply"))?
    }
}

impl Drop for Embedder {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            // Swap our `tx` with a fresh, never-used sender so the real one
            // drops here. The owner thread observes `Disconnected` on its
            // next `recv()` and exits its loop cleanly.
            let (dummy, _) = bounded::<Request>(1);
            drop(std::mem::replace(&mut self.tx, dummy));
            if let Err(panic) = handle.join() {
                warn!(?panic, "embedder thread panicked during shutdown");
            }
        }
    }
}

fn run_owner(
    backend: &'static LlamaBackend,
    model: LlamaModel,
    rx: crossbeam_channel::Receiver<Request>,
) {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(DEFAULT_N_CTX))
        .with_n_batch(DEFAULT_N_BATCH)
        .with_embeddings(true)
        // Qwen3-Embedding ships with pooling=Last in its GGUF metadata
        // (llama.cpp warns loudly if we override it); the model is trained
        // with the embedding read off the last token's hidden state, not a
        // mean over all positions. Match the model's default.
        .with_pooling_type(LlamaPoolingType::Last);

    let mut ctx = match model.new_context(backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            error!(error = %e, "embedder context init failed; draining queue with errors");
            // Drain pending requests so callers see a real error rather than
            // a generic channel-disconnect.
            while let Ok(req) = rx.recv() {
                let _ = req
                    .reply
                    .send(Err(anyhow!("embedder context init failed: {e}")));
            }
            return;
        }
    };

    debug!("embedder owner thread up");
    while let Ok(req) = rx.recv() {
        let result = embed_one(&model, &mut ctx, &req.text);
        // `send` fails only if the receiver was dropped (caller cancelled).
        if req.reply.send(result).is_err() {
            debug!("embed reply dropped (caller cancelled)");
        }
    }
    debug!("embedder owner thread shutting down");
}

fn embed_one(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    text: &str,
) -> Result<Vec<f32>> {
    // Reset KV cache so consecutive requests don't bleed into each other.
    // This is the same hygiene the spike-0a reranker uses.
    ctx.clear_kv_cache();

    let tokens = model
        .str_to_token(text, AddBos::Always)
        .context("tokenize")?;

    if tokens.is_empty() {
        return Err(anyhow!("tokenizer returned zero tokens for input"));
    }
    if tokens.len() > DEFAULT_N_CTX as usize {
        // Truncate rather than fail: a chunker that produced an oversized
        // chunk is a bug we want to discover via the resulting low recall,
        // not by aborting the whole embed pass.
        warn!(
            tokens = tokens.len(),
            max = DEFAULT_N_CTX,
            "chunk exceeds embedder context; truncating"
        );
    }
    let take = tokens.len().min(DEFAULT_N_CTX as usize);

    let mut batch = LlamaBatch::new(DEFAULT_N_BATCH as usize, 1);
    // `logits_all=true`: mark every token position as an output. Without
    // this, llama.cpp's embeddings-with-pooling path overrides our `false`
    // and logs the cosmetic "embeddings required but some input tokens were
    // not marked as outputs -> overriding" warning per decode. Passing
    // `true` upfront produces the same result silently — for an embedding
    // model with pooling enabled, all hidden states are already computed.
    batch
        .add_sequence(&tokens[..take], 0, true)
        .context("batch.add_sequence")?;

    ctx.decode(&mut batch).context("decode")?;

    let emb = ctx.embeddings_seq_ith(0).context("embeddings_seq_ith(0)")?;
    Ok(emb.to_vec())
}

#[cfg(test)]
mod tests {
    // Real-model tests live in markq-cli's e2e suite (gated on
    // MARKQ_TEST_MODEL). Pure unit tests here would need to mock
    // LlamaContext, which is `!Send` and has no trait abstraction — not
    // worth the scaffolding for a single embed path. Channel back-pressure /
    // dim-mismatch coverage lives in markq-index-lance and the embed e2e.
}
