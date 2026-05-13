//! end-to-end test: index a 10-file fixture, search, assert
//! known queries return known files at known ranks. This is the spec
//! check from `the plan` § — Tests.

use std::path::PathBuf;

use markq_cli::indexer::run_index;
use markq_core::{ChunkHit, Index};
use markq_index_lance::LanceIndex;

fn fixture_corpus() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/corpus")
}

async fn fresh_index() -> (LanceIndex, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chunks.lance");
    let idx = LanceIndex::open_or_create(&path).await.unwrap();
    (idx, dir)
}

fn first_path(hits: &[ChunkHit]) -> &str {
    &hits[0].path
}

#[tokio::test]
async fn index_corpus_then_search_known_queries() {
    let (idx, _tmp) = fresh_index().await;
    let report = run_index(&idx, &fixture_corpus()).await.unwrap();
    assert_eq!(report.files, 10, "expected 10 fixture files");
    assert!(report.chunks >= report.files, "at least one chunk per file");

    // Known query → known top-1 file. These pairs are chosen to be
    // unambiguous lexical matches — they should not regress under
    // tokenizer tweaks.
    let cases = &[
        ("HyDE query expansion", "hyde.md"),
        ("MCP server stdio", "mcp.md"),
        ("Reciprocal Rank Fusion", "rrf.md"),
        ("cross-encoder reranker", "rerank.md"),
        ("LanceDB FTS", "lance.md"),
    ];

    for (q, expected) in cases {
        let hits = idx.bm25(q, 5, None).await.unwrap();
        assert!(!hits.is_empty(), "query {q:?} returned no hits");
        let top = first_path(&hits);
        assert!(
            top.ends_with(expected),
            "query {q:?}: expected top hit to end with {expected:?}, got {top:?}"
        );
    }
}

#[tokio::test]
async fn empty_query_returns_empty() {
    let (idx, _tmp) = fresh_index().await;
    run_index(&idx, &fixture_corpus()).await.unwrap();
    let hits = idx.bm25("   ", 10, None).await.unwrap();
    assert!(hits.is_empty());
}

#[tokio::test]
async fn search_on_empty_index_returns_empty() {
    let (idx, _tmp) = fresh_index().await;
    let hits = idx.bm25("anything", 10, None).await.unwrap();
    assert!(hits.is_empty());
}

#[tokio::test]
async fn bm25_score_monotonicity_under_term_repetition() {
    // the plan: doubling a term frequency in a doc raises that doc's
    // score (or holds — never drops). We construct a 2-doc corpus where
    // doc A mentions "widgetron" once and doc B mentions it five times,
    // and assert B outscores A.
    let dir = tempfile::tempdir().unwrap();
    let corpus = dir.path().join("corpus");
    std::fs::create_dir_all(&corpus).unwrap();
    std::fs::write(
        corpus.join("a.md"),
        "# A\n\nThe widgetron is described once in this document.\n",
    )
    .unwrap();
    std::fs::write(
        corpus.join("b.md"),
        "# B\n\nwidgetron widgetron widgetron widgetron widgetron in this document.\n",
    )
    .unwrap();

    let ds = dir.path().join("chunks.lance");
    let idx = LanceIndex::open_or_create(&ds).await.unwrap();
    run_index(&idx, &corpus).await.unwrap();

    let hits = idx.bm25("widgetron", 5, None).await.unwrap();
    assert_eq!(hits.len(), 2);
    assert!(
        hits[0].path.ends_with("b.md"),
        "B (5x widgetron) should outrank A (1x); got {:?}",
        hits.iter().map(|h| (&h.path, h.score)).collect::<Vec<_>>()
    );
    assert!(hits[0].score >= hits[1].score);
}
