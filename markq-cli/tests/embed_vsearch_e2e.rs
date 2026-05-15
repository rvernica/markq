//! End-to-end embed + vector-search test, gated on `$MARKQ_TEST_MODEL`
//! pointing to a `Qwen3-Embedding-0.6B-Q8_0.gguf` file (the exact filename
//! `KnownModel::Qwen3Embedding06B.filename()` resolves to — `ensure_model`
//! looks for that name inside `$MARKQ_MODELS_DIR`, so a differently named
//! file would be invisible and trigger an HF download). The model is
//! multi-hundred-megabyte so we never download it from CI; run locally with:
//!
//!   MARKQ_TEST_MODEL=/path/to/Qwen3-Embedding-0.6B-Q8_0.gguf \
//!     cargo test --test embed_vsearch_e2e -- --ignored
//!
//! When the env var is set, the test exercises the full pipeline: index
//! the fixture corpus → embed → vsearch → re-embed (idempotent).

use std::path::PathBuf;

use markq_cli::indexer::run_index;
use markq_cli::{embedder_cmd, search, vsearch};
use markq_core::Index;
use markq_index_lance::LanceIndex;
use markq_inference::KnownModel;

fn fixture_corpus() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/corpus")
}

#[tokio::test]
#[ignore = "requires MARKQ_TEST_MODEL pointing at a Qwen3-Embedding GGUF"]
async fn embed_then_vsearch_end_to_end() {
    let model_path = match std::env::var("MARKQ_TEST_MODEL") {
        Ok(p) => PathBuf::from(p),
        Err(_) => return, // belt-and-braces; the #[ignore] above already gates this
    };
    assert!(
        model_path.exists(),
        "MARKQ_TEST_MODEL={} does not exist",
        model_path.display()
    );
    let expected_filename = KnownModel::Qwen3Embedding06B.filename();
    assert_eq!(
        model_path.file_name().and_then(|s| s.to_str()),
        Some(expected_filename),
        "MARKQ_TEST_MODEL must point at a file literally named {expected_filename}; \
         `ensure_model` joins MARKQ_MODELS_DIR with that hardcoded name and will \
         otherwise miss the local file and fall through to a Hugging Face download",
    );
    // Point markq's model cache at the directory holding the test model so
    // `ensure_model` reuses it instead of hitting Hugging Face.
    let parent = model_path
        .parent()
        .expect("MARKQ_TEST_MODEL must have a parent");
    std::env::set_var("MARKQ_MODELS_DIR", parent);

    let tmp = tempfile::tempdir().unwrap();
    let dataset = tmp.path().join("chunks.lance");
    let idx = LanceIndex::open_or_create(&dataset).await.unwrap();

    let index_report = run_index(&idx, &fixture_corpus()).await.unwrap();
    assert!(index_report.chunks > 0);

    let embed_report = embedder_cmd::run_embed(&idx).await.unwrap();
    assert_eq!(embed_report.rows, index_report.chunks as u64);

    let md = idx.metadata().await.unwrap();
    assert!(md.embedder_model.is_some());
    assert_eq!(md.embedder_dim, Some(embed_report.dim));

    let opts = search::SearchOptions {
        top_k: Some(5),
        min_score: None,
    };
    let hits = vsearch::run_vsearch(&idx, "how does reranking work", 5, &opts)
        .await
        .unwrap();
    assert!(!hits.is_empty(), "vsearch returned no hits");

    // Idempotent: a second embed pass has nothing to do.
    let embed_again = embedder_cmd::run_embed(&idx).await.unwrap();
    assert_eq!(embed_again.rows, 0);
}
