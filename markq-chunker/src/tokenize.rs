//! Token counting for the chunker. ships a deterministic, dependency-
//! free approximator (chars-per-token); will plug in the real Qwen
//! tokenizer once `markq-inference` lands. Keep this trait stable so the
//! swap is one line at the call site.

/// Counts tokens for a slice of source text. Implementors must be `Sync` so
/// the chunker can be called from rayon workers later.
pub trait Tokenize: Sync {
    fn count_tokens(&self, text: &str) -> usize;
}

/// Approximate tokenizer: ~4 chars per token, rounded up. Deterministic and
/// allocation-free. Good enough to drive packing decisions in — the
/// 900-token budget is itself a starting heuristic, not a hard contract.
pub struct ApproxTokenizer;

impl Tokenize for ApproxTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        let bytes = text.len();
        if bytes == 0 {
            0
        } else {
            bytes.div_ceil(4)
        }
    }
}
