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

use std::path::{Path, PathBuf};

use markq_cli::indexer::run_index;
use markq_cli::{embedder_cmd, search, vsearch};
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

/// Write a corpus of markdown files with *diverse* per-section vocabulary, so
/// the embeddings vary enough to drive real IVF training during `embed`. A
/// uniform corpus produces near-identical vectors and does not reproduce the
/// FTS-staleness regression this test guards. `common_term` is planted in every
/// section so a BM25 query for it should match (almost) every chunk.
fn write_diverse_corpus(dir: &Path, files: usize, sections: usize, common_term: &str) {
    const VOCAB: &[&str] = &[
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
        "kilo",
        "lima",
        "mike",
        "november",
        "oscar",
        "papa",
        "quebec",
        "romeo",
        "sierra",
        "tango",
        "uniform",
        "victor",
        "whiskey",
        "xray",
        "yankee",
        "zulu",
        "orbit",
        "photon",
        "quantum",
        "relay",
        "signal",
        "tensor",
        "vector",
        "wavelet",
        "boson",
        "gluon",
        "hadron",
        "lepton",
        "apple",
        "banana",
        "cherry",
        "date",
        "fig",
        "grape",
        "kiwi",
        "lemon",
        "mango",
        "nectarine",
    ];
    // Tiny deterministic LCG (no `rand` dep) so the corpus — and thus the
    // embeddings — are byte-for-byte reproducible across runs.
    let mut state: u64 = 0x9e3779b97f4a7c15;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    for f in 0..files {
        let mut body = format!("# Document {f}\n\n");
        for s in 0..sections {
            body.push_str(&format!("## Section {s}\n\n{common_term} "));
            for _ in 0..120 {
                body.push_str(VOCAB[next() % VOCAB.len()]);
                body.push(' ');
            }
            body.push_str("\n\n");
        }
        std::fs::write(dir.join(format!("doc_{f}.md")), body).unwrap();
    }
}

#[tokio::test]
#[ignore = "requires MARKQ_TEST_MODEL pointing at a Qwen3-Embedding GGUF"]
async fn embed_then_vsearch_end_to_end() {
    if setup_test_model().is_none() {
        return; // belt-and-braces; the #[ignore] above already gates this
    }

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

/// Regression: BM25 search must keep working after `embed`.
///
/// `embed` fills the embedding column via `merge_insert`, which rewrites every
/// row into fresh Lance fragments and tombstones the originals — leaving the
/// FTS index built at index time covering only dead rows. Before the fix,
/// `embed` rebuilt only the *vector* index, so a BM25 query that matched
/// hundreds of chunks before embedding collapsed to a handful afterward (and
/// the BM25 half of hybrid `query` silently degraded). `embed` now rebuilds the
/// FTS index too.
///
/// Reproducing this needs a corpus with *diverse* per-chunk text: uniform text
/// yields near-identical embeddings, degenerate IVF training, and the bug does
/// not surface. The chunker emits one chunk per heading section, so 60 files ×
/// 6 sections yields ~360 chunks (observed; the real-world corpus that first
/// exposed this was ~740) — comfortably clear of the `> 100` floor below. We
/// assert the BM25 hit count for a planted common term survives the embed pass.
#[tokio::test]
#[ignore = "requires MARKQ_TEST_MODEL pointing at a Qwen3-Embedding GGUF"]
async fn bm25_survives_embed() {
    if setup_test_model().is_none() {
        return;
    }

    let corpus = tempfile::tempdir().unwrap();
    let common = "github";
    write_diverse_corpus(corpus.path(), 60, 6, common);

    let tmp = tempfile::tempdir().unwrap();
    let dataset = tmp.path().join("chunks.lance");
    let idx = LanceIndex::open_or_create(&dataset).await.unwrap();

    let index_report = run_index(&idx, corpus.path()).await.unwrap();
    assert!(
        index_report.chunks > 100,
        "expected a multi-fragment corpus, got {} chunks",
        index_report.chunks
    );

    // `common` is planted in every section, so BM25 should match (almost)
    // every chunk both before and after embedding.
    let k = index_report.chunks + 10;
    let before = idx.bm25(common, k, None).await.unwrap().len();
    assert!(
        before > 100,
        "planted term should match most chunks before embed, got {before}"
    );

    let embed_report = embedder_cmd::run_embed(&idx).await.unwrap();
    assert_eq!(embed_report.rows, index_report.chunks as u64);

    let after = idx.bm25(common, k, None).await.unwrap().len();
    assert_eq!(
        after, before,
        "BM25 hit count collapsed across embed ({before} -> {after}); \
         the FTS index was left stale after merge_insert"
    );
}
