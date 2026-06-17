//! Token counting for the chunker. Ships a deterministic, dependency-free
//! approximator (chars-per-token); the real Qwen tokenizer can be plugged in
//! later once `markq-inference` lands. Keep this trait stable so that swap is
//! one line at the call site.

/// Counts tokens for a slice of source text. Implementors must be `Sync` so
/// the chunker can be called from rayon workers.
pub trait Tokenize: Sync {
    fn count_tokens(&self, text: &str) -> usize;
}

/// Approximate tokenizer: ~4 chars per token, rounded up. Deterministic and
/// allocation-free. Good enough to drive packing decisions — the 900-token
/// budget is itself a starting heuristic, not a hard contract.
///
/// Uses `chars().count()` rather than `text.len()` so non-ASCII content
/// (CJK, accented Latin, emoji) isn't over-counted by the 2–4x byte/char
/// ratio. A real Qwen tokenizer can be swapped in here later.
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
