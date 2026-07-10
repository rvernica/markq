//! End-to-end check: index + embed a small corpus, run a hybrid query,
//! assert the fused top-3 is stable and BM25-shape + vector-shape both
//! contribute to the trace.
//!
//! `#[ignore]` because it depends on the locally-cached Qwen3 embedder
//! model; CI does not download GGUFs. Run with:
//!   cargo test -p markq-cli --test hybrid_query -- --ignored

use std::fs;

use markq_cli::{indexer, query, search};
use markq_index_lance::LanceIndex;
use tempfile::TempDir;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn hybrid_query_fuses_lex_and_vec() {
    let dir = TempDir::new().expect("tempdir");
    let corpus = dir.path().join("corpus");
    fs::create_dir_all(&corpus).unwrap();
    fs::write(
        corpus.join("alpha.md"),
        "# Alpha\n\nBM25 reranking with hyphenated terms like tree-sitter.\n",
    )
    .unwrap();
    fs::write(
        corpus.join("beta.md"),
        "# Beta\n\nA discussion of semantic similarity in retrieval.\n",
    )
    .unwrap();
    fs::write(
        corpus.join("gamma.md"),
        "# Gamma\n\nOff-topic content about cooking pasta with garlic.\n",
    )
    .unwrap();

    let dataset = dir.path().join("chunks.lance");
    let idx = LanceIndex::open_or_create(&dataset)
        .await
        .expect("open dataset");
    indexer::run_index(&idx, &corpus).await.expect("index");

    // Embed (will load the model from ~/.cache/markq/models/).
    let embed_report = markq_cli::embedder_cmd::run_embed(&idx)
        .await
        .expect("embed");
    assert!(embed_report.rows >= 3);

    // LanceDB reads a snapshot at open time; the BM25 / FTS index built
    // during `run_index` (and the embeddings written during `run_embed`)
    // are not visible to the existing handle. Reopen to pick them up.
    let idx = LanceIndex::open(&dataset).await.expect("reopen");

    let opts = search::SearchOptions {
        top_k: Some(3),
        min_score: None,
    };
    let outcome = query::run_query(&idx, "semantic similarity retrieval", 3, &opts, true, false)
        .await
        .expect("run_query");

    assert!(!outcome.hits.is_empty(), "expected at least one fused hit");
    let trace = outcome.explain.expect("explain trace");
    assert!(trace.bm25_hits >= 1, "bm25 should match some terms");
    // Each of the 3 fixture files produces exactly one chunk; the vector
    // KNN returns every chunk in the dataset because k_pre exceeds 3.
    assert_eq!(
        trace.vector_hits, 3,
        "vector should return every chunk in the 3-doc corpus",
    );

    // The top hit should be `beta.md` — it's the only doc with both the
    // BM25 keyword "retrieval" and the semantic signal for "similarity".
    let top = &outcome.hits[0];
    assert!(
        top.path.ends_with("beta.md"),
        "top hit was {}, expected beta.md",
        top.path
    );
}
