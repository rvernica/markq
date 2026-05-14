//! Token counting for the chunker. Phase 2 ships a deterministic, dependency-
//! free approximator (chars-per-token); Phase 4 will plug in the real Qwen
//! tokenizer once `markq-inference` lands. Keep this trait stable so the
//! Phase 4 swap is one line at the call site.

/// Counts tokens for a slice of source text. Implementors must be `Sync` so
/// the chunker can be called from rayon workers later (Phase 3).
pub trait Tokenize: Sync {
    fn count_tokens(&self, text: &str) -> usize;
}

/// Approximate tokenizer: ~4 chars per token, rounded up. Deterministic and
/// allocation-free. Good enough to drive packing decisions in Phase 2 — the
/// 900-token budget is itself a starting heuristic, not a hard contract.
///
/// Uses `chars().count()` rather than `text.len()` so non-ASCII content
/// (CJK, accented Latin, emoji) isn't over-counted by the 2–4x byte/char
/// ratio. Phase 4 swaps this for the real Qwen tokenizer.
pub struct ApproxTokenizer;

impl Tokenize for ApproxTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        let chars = text.chars().count();
        if chars == 0 {
            0
        } else {
            chars.div_ceil(4)
        }
    }
}
