//! Locks the `--json` output contract shared by `search`, `vsearch`, and
//! `query`: every element of the emitted array must parse as
//! `markq_cli::candidate::Candidate` with a non-empty `id` and non-empty
//! `text`. All three commands format `--json` through the single
//! `search::write_json` function, so this test pins that shared contract
//! rather than re-deriving it three times.
//!
//! The `search` (BM25) arm needs no model and runs unconditionally. The
//! `vsearch` / `query` arms need embeddings, so they are gated behind
//! `$MARKQ_TEST_MODEL` exactly like `embed_vsearch_e2e.rs`:
//!
//!   MARKQ_TEST_MODEL=/path/to/Qwen3-Embedding-0.6B-Q8_0.gguf \
//!     cargo test -p markq-cli --test json_contract -- --ignored

use std::path::PathBuf;

use markq_cli::candidate::Candidate;
use markq_cli::indexer::run_index;
use markq_cli::{embedder_cmd, query, search, vsearch};
use markq_core::Index;
use markq_index_lance::LanceIndex;
use markq_inference::KnownModel;

fn fixture_corpus() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/corpus")
}

/// Resolve `$MARKQ_TEST_MODEL`, assert it points at the expected filename, and
/// wire `MARKQ_MODELS_DIR` so `ensure_model` reuses it instead of downloading.
/// Returns `None` (so the caller can early-return) when the var is unset.
fn setup_test_model() -> Option<PathBuf> {
    let model_path = PathBuf::from(std::env::var("MARKQ_TEST_MODEL").ok()?);
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
    let parent = model_path
        .parent()
        .expect("MARKQ_TEST_MODEL must have a parent");
    std::env::set_var("MARKQ_MODELS_DIR", parent);
    Some(model_path)
}

/// Parse `buf` as a JSON array of `Candidate` and assert every element has a
/// non-empty `id` and non-empty `text`. Returns the parsed candidates so
/// callers can also assert non-emptiness of the array itself.
fn assert_candidate_contract(buf: &[u8], label: &str) -> Vec<Candidate> {
    let candidates: Vec<Candidate> =
        serde_json::from_slice(buf).unwrap_or_else(|e| panic!("{label}: invalid JSON: {e}"));
    assert!(!candidates.is_empty(), "{label}: expected non-empty array");
    for c in &candidates {
        assert!(!c.id.is_empty(), "{label}: candidate has empty id: {c:?}");
        assert!(
            !c.text.is_empty(),
            "{label}: candidate has empty text: {c:?}"
        );
    }
    candidates
}

#[tokio::test]
async fn search_json_output_is_candidate_array() {
    let tmp = tempfile::tempdir().unwrap();
    let dataset = tmp.path().join("chunks.lance");
    let idx = LanceIndex::open_or_create(&dataset).await.unwrap();

    let index_report = run_index(&idx, &fixture_corpus()).await.unwrap();
    assert!(index_report.chunks > 0);

    let opts = search::SearchOptions {
        top_k: Some(10),
        min_score: None,
    };
    let raw = idx.bm25("markq", 10, None).await.unwrap();
    let hits = search::apply_filters(raw, &opts);
    assert!(!hits.is_empty(), "bm25 search returned no hits");

    let mut buf = Vec::new();
    search::write_results(&mut buf, &hits, search::Format::Json).unwrap();

    assert_candidate_contract(&buf, "search");
}

#[tokio::test]
#[ignore = "requires MARKQ_TEST_MODEL pointing at a Qwen3-Embedding GGUF"]
async fn vsearch_json_output_is_candidate_array() {
    if setup_test_model().is_none() {
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let dataset = tmp.path().join("chunks.lance");
    let idx = LanceIndex::open_or_create(&dataset).await.unwrap();

    let index_report = run_index(&idx, &fixture_corpus()).await.unwrap();
    assert!(index_report.chunks > 0);

    let embed_report = embedder_cmd::run_embed(&idx).await.unwrap();
    assert_eq!(embed_report.rows, index_report.chunks as u64);

    let opts = search::SearchOptions {
        top_k: Some(5),
        min_score: None,
    };
    let hits = vsearch::run_vsearch(&idx, "how does reranking work", 5, &opts)
        .await
        .unwrap();
    assert!(!hits.is_empty(), "vsearch returned no hits");

    let mut buf = Vec::new();
    search::write_results(&mut buf, &hits, search::Format::Json).unwrap();

    assert_candidate_contract(&buf, "vsearch");
}

#[tokio::test]
#[ignore = "requires MARKQ_TEST_MODEL pointing at a Qwen3-Embedding GGUF"]
async fn query_json_output_is_candidate_array() {
    if setup_test_model().is_none() {
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let dataset = tmp.path().join("chunks.lance");
    let idx = LanceIndex::open_or_create(&dataset).await.unwrap();

    let index_report = run_index(&idx, &fixture_corpus()).await.unwrap();
    assert!(index_report.chunks > 0);

    let embed_report = embedder_cmd::run_embed(&idx).await.unwrap();
    assert_eq!(embed_report.rows, index_report.chunks as u64);

    let opts = search::SearchOptions {
        top_k: Some(5),
        min_score: None,
    };
    let outcome = query::run_query(&idx, "how does reranking work", 5, &opts, false)
        .await
        .unwrap();
    assert!(!outcome.hits.is_empty(), "query returned no hits");

    let mut buf = Vec::new();
    search::write_results(&mut buf, &outcome.hits, search::Format::Json).unwrap();

    assert_candidate_contract(&buf, "query");
}
