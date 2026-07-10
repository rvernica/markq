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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_matches_model_card_shape_exactly() {
        let rendered = render_prompt(DEFAULT_INSTRUCTION, "sample query", "sample document");

        let expected = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: sample query\n<Document>: sample document<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

        assert_eq!(rendered, expected);
    }
}
