//! Reranker prompt rendering. The Qwen3-Reranker-0.6B model is prompted as a
//! yes/no judge: given an instruction, a query, and a candidate document, it
//! decides whether the document satisfies the query under that instruction.
//! This module holds the pure prompt-rendering piece; the owner thread that
//! loads the model and turns the yes/no logits into a score follows the same
//! shape as `embedder.rs` and lands in a later change.

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
}
