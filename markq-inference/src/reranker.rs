//! Reranker: the Qwen3-Reranker-0.6B model is prompted as a yes/no judge —
//! given an instruction, a query, and a candidate document, it decides whether
//! the document satisfies the query under that instruction. This module holds
//! both the pure prompt-rendering piece and the owner thread that loads the
//! model and turns its next-token logits into a relevance score.
//!
//! Scoring drives the model as an ordinary causal LM (no embedding pooling):
//! we decode the rendered chat-template prompt, read the vocabulary logits at
//! the final position, and take a two-way softmax over just the "yes" and "no"
//! token logits. The result is `P("yes")` in `[0, 1]`.
//!
//! Ownership mirrors `embedder.rs` exactly: one owner thread holds the
//! `LlamaModel` + `LlamaContext` (the latter is `!Send`), fed by a bounded
//! `crossbeam-channel`, replying over a `oneshot`. Dropping the handle shuts
//! the thread down cleanly.

use std::num::NonZeroU32;
use std::path::Path;
use std::thread::{self, JoinHandle};

use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{bounded, Sender};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
#[allow(deprecated)]
use llama_cpp_2::model::Special;
use llama_cpp_2::model::{AddBos, LlamaModel};
use tokio::sync::oneshot;
use tracing::{debug, error, warn};

use crate::backend;

/// Bounded queue capacity. Producers block once this many score requests are
/// in flight; the owner thread drains them as fast as inference allows.
const CHANNEL_CAPACITY: usize = 256;

/// llama.cpp context size. Matches the embedder: markq chunks target ~900
/// tokens, and 2048 leaves headroom for the chat template, query, and BOS.
const DEFAULT_N_CTX: u32 = 2048;

/// llama.cpp logical batch size. Sized so a worst-case prompt fits in one
/// decode.
const DEFAULT_N_BATCH: u32 = 2048;

/// Safety margin (in tokens) subtracted from the document token budget, to
/// absorb tokenization boundary effects (e.g. a token that merges across the
/// truncation point tokenizing differently once the document is cut and
/// re-rendered into the full prompt).
const TRUNCATION_MARGIN: usize = 8;

struct Request {
    instruction: String,
    query: String,
    document: String,
    reply: oneshot::Sender<Result<f32>>,
}

/// Handle to a running reranker. Drop to shut down the owner thread (the
/// channel sender is dropped, the owner observes `Disconnected`, exits).
pub struct Reranker {
    tx: Sender<Request>,
    /// Joined on `Drop`; held in `Option` so `drop` can take it.
    handle: Option<JoinHandle<()>>,
}

impl Reranker {
    /// Load a reranker GGUF and spawn the owner thread.
    ///
    /// `n_gpu_layers = 999` offloads every layer when the crate was built with
    /// the `vulkan` / `cuda` feature; on a CPU-only build the value is ignored.
    pub fn load(path: &Path, n_gpu_layers: u32) -> Result<Reranker> {
        let backend = backend::shared()?;
        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(backend, path, &model_params)
            .with_context(|| format!("load_from_file {}", path.display()))?;

        let (tx, rx) = bounded::<Request>(CHANNEL_CAPACITY);
        let handle = thread::Builder::new()
            .name("markq-reranker".to_string())
            .spawn(move || run_owner(backend, model, rx))
            .context("spawn reranker thread")?;

        Ok(Reranker {
            tx,
            handle: Some(handle),
        })
    }

    /// Score one (query, document) pair for relevance. `instruction` defaults
    /// to [`DEFAULT_INSTRUCTION`] when `None`. Returns `P("yes")` in `[0, 1]`:
    /// higher means more relevant. Awaits the owner thread's reply, blocking
    /// (via the bounded channel) under back-pressure.
    ///
    /// The raw (instruction, query, document) pieces are sent as-is; prompt
    /// rendering and any document truncation happen on the owner thread,
    /// which is the only place the model (and thus its tokenizer) lives.
    pub async fn score(
        &self,
        query: &str,
        document: &str,
        instruction: Option<&str>,
    ) -> Result<f32> {
        let instruction = instruction
            .map(str::to_string)
            .unwrap_or_else(|| DEFAULT_INSTRUCTION.to_string());

        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Request {
                instruction,
                query: query.to_string(),
                document: document.to_string(),
                reply,
            })
            .map_err(|_| anyhow!("reranker thread is gone"))?;
        rx.await.map_err(|_| anyhow!("reranker dropped reply"))?
    }
}

impl Drop for Reranker {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            // Swap our `tx` with a fresh, never-used sender so the real one
            // drops here. The owner observes `Disconnected` on its next
            // `recv()` and exits its loop cleanly.
            let (dummy, _) = bounded::<Request>(1);
            drop(std::mem::replace(&mut self.tx, dummy));
            if let Err(panic) = handle.join() {
                warn!(?panic, "reranker thread panicked during shutdown");
            }
        }
    }
}

/// The single-token vocab ids for the "yes" / "no" answers, resolved once from
/// the loaded model's own tokenizer (never hardcoded).
struct YesNo {
    yes: llama_cpp_2::token::LlamaToken,
    no: llama_cpp_2::token::LlamaToken,
}

/// Resolve the "yes"/"no" answer token ids from this model's tokenizer. The
/// prompt ends right where the assistant's one-word verdict begins, so the
/// standalone (no leading space) forms are what the model emits. When a word
/// tokenizes to several pieces we take the first, which carries the lexical
/// content.
fn resolve_yes_no(model: &LlamaModel) -> Result<YesNo> {
    let first = |word: &str| -> Result<llama_cpp_2::token::LlamaToken> {
        let toks = model
            .str_to_token(word, AddBos::Never)
            .with_context(|| format!("tokenize {word:?}"))?;
        if toks.len() != 1 {
            warn!(
                word,
                tokens = toks.len(),
                "reranker yes/no word tokenized to more than one token; using the first"
            );
        }
        toks.first()
            .copied()
            .ok_or_else(|| anyhow!("tokenizer returned no tokens for {word:?}"))
    };
    let yes = first("yes")?;
    let no = first("no")?;
    debug!(yes = yes.0, no = no.0, "resolved reranker yes/no token ids");
    Ok(YesNo { yes, no })
}

fn run_owner(
    backend: &'static LlamaBackend,
    model: LlamaModel,
    rx: crossbeam_channel::Receiver<Request>,
) {
    // Ordinary causal-LM logits: NO embeddings, NO pooling (unlike the
    // embedder). We need the full vocab distribution at the final token.
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(DEFAULT_N_CTX))
        .with_n_batch(DEFAULT_N_BATCH);

    let yes_no = match resolve_yes_no(&model) {
        Ok(y) => y,
        Err(e) => {
            error!(error = %e, "reranker yes/no token resolution failed; draining queue with errors");
            while let Ok(req) = rx.recv() {
                let _ = req
                    .reply
                    .send(Err(anyhow!("reranker yes/no token resolution failed: {e}")));
            }
            return;
        }
    };

    let mut ctx = match model.new_context(backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            error!(error = %e, "reranker context init failed; draining queue with errors");
            while let Ok(req) = rx.recv() {
                let _ = req
                    .reply
                    .send(Err(anyhow!("reranker context init failed: {e}")));
            }
            return;
        }
    };

    debug!("reranker owner thread up");
    while let Ok(req) = rx.recv() {
        let result = score_one(
            &model,
            &mut ctx,
            &yes_no,
            &req.instruction,
            &req.query,
            &req.document,
        );
        if req.reply.send(result).is_err() {
            debug!("rerank reply dropped (caller cancelled)");
        }
    }
    debug!("reranker owner thread shutting down");
}

fn score_one(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    yes_no: &YesNo,
    instruction: &str,
    query: &str,
    document: &str,
) -> Result<f32> {
    // Reset KV cache so consecutive requests don't bleed into each other.
    ctx.clear_kv_cache();

    // The chat template puts the assistant/verdict scaffolding
    // (`<|im_start|>assistant\n<think>\n\n</think>\n\n`) *after* the document,
    // so truncating the rendered prompt from the front — as a naive
    // "too-long text" truncation would — discards exactly that scaffolding;
    // `get_logits_ith(last)` would then read a position in the middle of the
    // document, making the score meaningless rather than merely degraded.
    // Instead, bound the *document* so the template prefix (system +
    // instruct + query) and the verdict suffix are always preserved intact.
    let overhead = model
        .str_to_token(&render_prompt(instruction, query, ""), AddBos::Always)
        .context("tokenize template overhead")?
        .len();
    // The query/instruction template overhead is shared by every candidate in
    // a rerank batch. If it alone leaves no room for a document (plus the
    // truncation margin) within the context window, there is no safe document
    // budget to fall back to: truncating the document to (near-)empty would
    // still leave a rendered prompt that saturates or exceeds `DEFAULT_N_CTX`,
    // and the belt-and-braces `take` clamp below would then front-truncate the
    // prompt itself, silently discarding the verdict-scaffolding suffix and
    // scoring a meaningless mid-prompt position. Fail loudly instead.
    if overhead + TRUNCATION_MARGIN >= DEFAULT_N_CTX as usize {
        return Err(anyhow!(
            "query/instruction too long to rerank ({overhead} tokens leaves no document budget within the {DEFAULT_N_CTX}-token context after a {TRUNCATION_MARGIN}-token margin); shorten the query"
        ));
    }
    let doc_budget = (DEFAULT_N_CTX as usize) - (overhead + TRUNCATION_MARGIN);

    let doc_tokens = model
        .str_to_token(document, AddBos::Never)
        .context("tokenize document")?;

    let document = if doc_tokens.len() > doc_budget {
        warn!(
            doc_tokens = doc_tokens.len(),
            budget = doc_budget,
            "rerank document exceeds context budget; truncating document"
        );
        #[allow(deprecated)]
        let truncated = model
            .tokens_to_str(&doc_tokens[..doc_budget], Special::Plaintext)
            .context("detokenize truncated document")?;
        truncated
    } else {
        document.to_string()
    };

    let prompt = render_prompt(instruction, query, &document);
    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .context("tokenize")?;
    if tokens.is_empty() {
        return Err(anyhow!("tokenizer returned zero tokens for prompt"));
    }
    // Belt-and-braces: the budget above should already guarantee this fits,
    // but never index outside the context window.
    let take = tokens.len().min(DEFAULT_N_CTX as usize);

    let mut batch = LlamaBatch::new(DEFAULT_N_BATCH as usize, 1);
    // `add_sequence(.., false)` marks only the final token as an output, which
    // is exactly the position whose next-token distribution we score.
    batch
        .add_sequence(&tokens[..take], 0, false)
        .context("batch.add_sequence")?;

    ctx.decode(&mut batch).context("decode")?;

    // Vocab logits at the final position.
    let last = i32::try_from(take - 1).context("final token index overflows i32")?;
    let logits = ctx.get_logits_ith(last);

    let yes_idx = yes_no.yes.0 as usize;
    let no_idx = yes_no.no.0 as usize;
    let l_yes = *logits
        .get(yes_idx)
        .ok_or_else(|| anyhow!("yes token id {yes_idx} out of logits range"))?;
    let l_no = *logits
        .get(no_idx)
        .ok_or_else(|| anyhow!("no token id {no_idx} out of logits range"))?;

    Ok(yes_no_softmax(l_yes, l_no))
}

/// Two-way softmax over the "yes"/"no" logits, computed in a numerically
/// stable way (subtract the max before exponentiating). Returns `P("yes")` in
/// `[0, 1]`. Pure and side-effect free: no tokenization, no model access — the
/// full scoring pipeline's only non-trivial arithmetic, so it's the piece
/// worth unit testing directly (the rest of `score_one` needs a real model
/// tokenizer and is exercised only by the gated integration test).
fn yes_no_softmax(l_yes: f32, l_no: f32) -> f32 {
    let m = l_yes.max(l_no);
    let e_yes = (l_yes - m).exp();
    let e_no = (l_no - m).exp();
    e_yes / (e_yes + e_no)
}

/// Canonical Qwen3-Reranker retrieval instruction, used when the caller
/// doesn't supply one of their own (e.g. the CLI's `--instruction` flag left
/// at its default).
pub const DEFAULT_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

/// Render the Qwen3-Reranker chat template for a single (instruction, query,
/// document) triple. Pure and side-effect free: no tokenization, no model
/// access — just string assembly, so it's trivial to unit test byte-for-byte.
pub fn render_prompt(instruction: &str, query: &str, document: &str) -> String {
    format!(
        "<|im_start|>system\n\
         Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n\
         <|im_start|>user\n\
         <Instruct>: {instruction}\n\
         <Query>: {query}\n\
         <Document>: {document}<|im_end|>\n\
         <|im_start|>assistant\n\
         <think>\n\
         \n\
         </think>\n\
         \n"
    )
}

/// Order candidate indices by relevance score, descending, breaking ties by
/// ascending original input index. Pure and deterministic: no I/O, no model —
/// a stable sort over index-enumerated input naturally keeps equal-scored
/// candidates in their original order, which is what makes reruns
/// byte-identical.
///
/// `top_k` optionally truncates the sorted output to its first `k` entries.
/// `top_k >= scores.len()` returns every index (no padding, no error);
/// `top_k == Some(0)` returns an empty vector; `top_k == None` returns all
/// indices in sorted order.
pub fn order_by_relevance(scores: &[f32], top_k: Option<usize>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();

    // Stable sort: equal scores retain their relative (ascending index)
    // order because `order` starts in ascending index order.
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));

    if let Some(k) = top_k {
        order.truncate(k);
    }

    order
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real-model scoring test, gated on `$MARKQ_TEST_RERANK_MODEL` pointing at
    /// the reranker GGUF (a multi-hundred-MB file that must never run in CI).
    /// Confirms the yes/no scorer separates a clearly relevant (query,
    /// document) pair from an irrelevant one and that both scores are valid
    /// probabilities in `[0, 1]`.
    #[tokio::test]
    #[ignore = "requires MARKQ_TEST_RERANK_MODEL pointing at a Qwen3-Reranker GGUF"]
    async fn scorer_separates_relevant_from_irrelevant() {
        let path = match std::env::var("MARKQ_TEST_RERANK_MODEL") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => {
                eprintln!("MARKQ_TEST_RERANK_MODEL unset; skipping");
                return;
            }
        };
        assert!(
            path.exists(),
            "MARKQ_TEST_RERANK_MODEL={} does not exist",
            path.display()
        );

        let reranker =
            Reranker::load(&path, crate::default_n_gpu_layers()).expect("load reranker GGUF");

        let query = "What is the capital of France?";
        let pos = reranker
            .score(
                query,
                "Paris is the capital and most populous city of France.",
                None,
            )
            .await
            .expect("score positive pair");
        let neg = reranker
            .score(
                query,
                "Soccer is a popular sport played around the world.",
                None,
            )
            .await
            .expect("score negative pair");

        eprintln!("pos={pos:.6}  neg={neg:.6}  Δ={:+.6}", pos - neg);

        assert!(
            (0.0..=1.0).contains(&pos),
            "positive score {pos} out of [0,1]"
        );
        assert!(
            (0.0..=1.0).contains(&neg),
            "negative score {neg} out of [0,1]"
        );
        assert!(
            pos > neg,
            "expected relevant pair to score higher: pos={pos}, neg={neg}"
        );
    }

    #[test]
    fn template_matches_model_card_shape_exactly() {
        let rendered = render_prompt(DEFAULT_INSTRUCTION, "sample query", "sample document");

        let expected = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: sample query\n<Document>: sample document<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

        assert_eq!(rendered, expected);
    }

    #[test]
    fn order_by_relevance_breaks_ties_by_ascending_input_index() {
        let scores = [0.9, 0.5, 0.9, 0.5];

        let order = order_by_relevance(&scores, None);

        assert_eq!(order, vec![0, 2, 1, 3]);
    }

    #[test]
    fn order_by_relevance_truncates_to_top_k() {
        let scores = [0.9, 0.5, 0.9, 0.5];

        let order = order_by_relevance(&scores, Some(2));

        assert_eq!(order, vec![0, 2]);
    }

    #[test]
    fn order_by_relevance_top_k_larger_than_len_returns_all() {
        let scores = [0.9, 0.5, 0.9, 0.5];

        let order = order_by_relevance(&scores, Some(99));

        assert_eq!(order, vec![0, 2, 1, 3]);
    }

    #[test]
    fn order_by_relevance_top_k_zero_returns_empty() {
        let scores = [0.9, 0.5, 0.9, 0.5];

        let order = order_by_relevance(&scores, Some(0));

        assert_eq!(order, Vec::<usize>::new());
    }

    #[test]
    fn yes_no_softmax_equal_logits_is_exactly_half() {
        assert_eq!(yes_no_softmax(0.0, 0.0), 0.5);
        assert_eq!(yes_no_softmax(3.7, 3.7), 0.5);
        assert_eq!(yes_no_softmax(-12.0, -12.0), 0.5);
    }

    #[test]
    fn yes_no_softmax_is_monotonic_in_each_logit() {
        assert!(yes_no_softmax(2.0, 0.0) > 0.5, "higher yes-logit -> > 0.5");
        assert!(yes_no_softmax(0.0, 2.0) < 0.5, "higher no-logit -> < 0.5");
    }

    #[test]
    fn yes_no_softmax_is_stable_at_extreme_logits() {
        let strongly_yes = yes_no_softmax(100.0, -100.0);
        assert!(strongly_yes.is_finite(), "must not be NaN/inf");
        assert!((0.0..=1.0).contains(&strongly_yes));
        assert!(strongly_yes > 0.999);

        let strongly_no = yes_no_softmax(-100.0, 100.0);
        assert!(strongly_no.is_finite(), "must not be NaN/inf");
        assert!((0.0..=1.0).contains(&strongly_no));
        assert!(strongly_no < 0.001);
    }

    #[test]
    fn yes_no_softmax_always_within_unit_interval() {
        for (l_yes, l_no) in [
            (0.0, 0.0),
            (1.5, -3.2),
            (-1.5, 3.2),
            (50.0, 50.0),
            (-50.0, -50.0),
            (1e6, -1e6),
        ] {
            let p = yes_no_softmax(l_yes, l_no);
            assert!(
                (0.0..=1.0).contains(&p),
                "yes_no_softmax({l_yes}, {l_no}) = {p} out of [0,1]"
            );
        }
    }
}
