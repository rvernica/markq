//! End-to-end `run_rerank` test, gated on `$MARKQ_TEST_RERANK_MODEL` pointing
//! at a `qwen3-reranker-0.6b-q8_0.gguf` file (the exact filename
//! `KnownModel::Qwen3Reranker06B.filename()` resolves to — `ensure_model`
//! looks for that name inside `$MARKQ_MODELS_DIR`, so a differently named
//! file would be invisible and trigger a Hugging Face download). The model is
//! multi-hundred-megabyte so we never download it from CI; run locally with:
//!
//!   MARKQ_TEST_RERANK_MODEL=/path/to/qwen3-reranker-0.6b-q8_0.gguf \
//!     cargo test -p markq-cli --test rerank -- --ignored --nocapture

use std::path::PathBuf;

use markq_cli::rerank::run_rerank;
use markq_inference::KnownModel;

/// Resolve `$MARKQ_TEST_RERANK_MODEL`, assert it points at the expected
/// filename, and wire `MARKQ_MODELS_DIR` so `ensure_model` reuses it instead
/// of downloading. Returns `None` (so the caller can early-return) when the
/// var is unset.
fn setup_test_model() -> Option<PathBuf> {
    let model_path = PathBuf::from(std::env::var("MARKQ_TEST_RERANK_MODEL").ok()?);
    assert!(
        model_path.exists(),
        "MARKQ_TEST_RERANK_MODEL={} does not exist",
        model_path.display()
    );
    let expected_filename = KnownModel::Qwen3Reranker06B.filename();
    assert_eq!(
        model_path.file_name().and_then(|s| s.to_str()),
        Some(expected_filename),
        "MARKQ_TEST_RERANK_MODEL must point at a file literally named {expected_filename}; \
         `ensure_model` joins MARKQ_MODELS_DIR with that hardcoded name and will \
         otherwise miss the local file and fall through to a Hugging Face download",
    );
    let parent = model_path
        .parent()
        .expect("MARKQ_TEST_RERANK_MODEL must have a parent");
    std::env::set_var("MARKQ_MODELS_DIR", parent);
    Some(model_path)
}

#[tokio::test]
#[ignore = "requires MARKQ_TEST_RERANK_MODEL pointing at a Qwen3-Reranker GGUF"]
async fn reorders_most_relevant_candidate_to_rank_one() {
    if setup_test_model().is_none() {
        return; // belt-and-braces; the #[ignore] above already gates this
    }

    let query = "What is the capital of France?";
    let input = serde_json::json!([
        {"id": "a", "text": "Soccer is a popular sport played around the world.", "score": 0.5, "collection": "docs"},
        {"id": "b", "text": "The stock market fluctuated wildly this week.", "score": 0.4},
        {"id": "c", "text": "Paris is the capital and most populous city of France.", "score": 0.1},
    ]);
    let bytes = serde_json::to_vec(&input).unwrap();

    let mut buf = Vec::new();
    run_rerank(
        bytes.as_slice(),
        &mut buf,
        query,
        Some(2),
        None,
        /* json */ true,
    )
    .await
    .expect("run_rerank");

    let out: serde_json::Value = serde_json::from_slice(&buf).expect("valid JSON output");
    let arr = out.as_array().expect("output is a JSON array");

    // top_k = 2 truncation holds.
    assert_eq!(arr.len(), 2, "expected top-k=2 truncation, got {arr:?}");

    // The Paris candidate (placed last in the input) must be reordered to
    // rank 1.
    assert_eq!(arr[0]["id"], "c", "expected Paris candidate at rank 1");
    assert_eq!(arr[0]["rank"], 1);

    // Every element carries rerank_score + rank, and ids are preserved from
    // the input.
    for el in arr {
        assert!(el.get("rerank_score").is_some(), "missing rerank_score");
        assert!(el.get("rank").is_some(), "missing rank");
        assert!(el.get("id").is_some(), "missing id");
        let score = el["rerank_score"].as_f64().unwrap();
        assert!(
            (0.0..=1.0).contains(&score),
            "rerank_score {score} out of [0,1]"
        );
    }

    // Passthrough fields from the original candidate survive reordering
    // verbatim.
    let paris = arr.iter().find(|el| el["id"] == "c").unwrap();
    assert_eq!(paris["score"], serde_json::json!(0.1));
}
