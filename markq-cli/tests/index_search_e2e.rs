//! End-to-end test: index a 10-file fixture, search, assert known queries
//! return known files at known ranks.

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

/// Write a throwaway corpus of `(relative_path, body)` files into a fresh
/// tempdir and return it. Kept separate from the `.lance` dataset dir so the
/// indexer's walk never sees the dataset.
fn corpus_with(files: &[(&str, &str)]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for (name, body) in files {
        let p = dir.path().join(name);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(p, body).unwrap();
    }
    dir
}

/// Count distinct chunk ids returned for a query — duplicates inflate the
/// difference between `hits.len()` and the distinct-id count.
fn distinct_ids(hits: &[ChunkHit]) -> usize {
    hits.iter()
        .map(|h| h.id.as_str())
        .collect::<std::collections::HashSet<_>>()
        .len()
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
async fn reindex_unchanged_corpus_is_idempotent() {
    let (idx, _tmp) = fresh_index().await;
    let corpus = corpus_with(&[
        ("a.md", "# A\n\nalpha content about widgets\n"),
        ("b.md", "# B\n\nbravo content about gadgets\n"),
    ]);

    run_index(&idx, corpus.path()).await.unwrap();
    let rows_after_first = idx.count_rows().await.unwrap();

    run_index(&idx, corpus.path()).await.unwrap();
    let rows_after_second = idx.count_rows().await.unwrap();

    assert_eq!(
        rows_after_second, rows_after_first,
        "re-indexing an unchanged corpus must not add rows"
    );

    let hits = idx.bm25("content", 10, None).await.unwrap();
    assert_eq!(
        distinct_ids(&hits),
        hits.len(),
        "results contain duplicate chunk ids after re-index: {:?}",
        hits.iter()
            .map(|h| (&h.path, h.chunk_index))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn reindex_replaces_edited_file() {
    let (idx, _tmp) = fresh_index().await;
    let corpus = corpus_with(&[("note.md", "# Note\n\nThe term zephyrquux appears here.\n")]);
    let file = corpus.path().join("note.md");

    run_index(&idx, corpus.path()).await.unwrap();
    assert!(
        !idx.bm25("zephyrquux", 10, None).await.unwrap().is_empty(),
        "old term should be present after first index"
    );

    // Edit the file: drop the old unique term, introduce a new one.
    std::fs::write(&file, "# Note\n\nNow it mentions plingwobble instead.\n").unwrap();
    run_index(&idx, corpus.path()).await.unwrap();

    let old = idx.bm25("zephyrquux", 10, None).await.unwrap();
    assert!(
        old.is_empty(),
        "edited-away term must not remain searchable (orphaned chunks): {old:?}"
    );
    let new = idx.bm25("plingwobble", 10, None).await.unwrap();
    assert!(
        !new.is_empty(),
        "new content must be searchable after re-index"
    );
    assert_eq!(distinct_ids(&new), new.len(), "no duplicate ids after edit");
}

#[tokio::test]
async fn reindex_prunes_removed_file() {
    let (idx, _tmp) = fresh_index().await;
    let corpus = corpus_with(&[
        (
            "keep.md",
            "# Keep\n\nThis file has staysearchable content.\n",
        ),
        (
            "gone.md",
            "# Gone\n\nThis file has vanishingterm content.\n",
        ),
    ]);
    run_index(&idx, corpus.path()).await.unwrap();
    assert!(
        !idx.bm25("vanishingterm", 10, None)
            .await
            .unwrap()
            .is_empty(),
        "both files searchable after first index"
    );

    // Remove one file from disk, then re-index.
    std::fs::remove_file(corpus.path().join("gone.md")).unwrap();
    run_index(&idx, corpus.path()).await.unwrap();

    let gone = idx.bm25("vanishingterm", 10, None).await.unwrap();
    assert!(
        gone.is_empty(),
        "chunks of a file removed from disk must be pruned: {gone:?}"
    );
    assert!(
        !idx.bm25("staysearchable", 10, None)
            .await
            .unwrap()
            .is_empty(),
        "surviving file must remain searchable"
    );
}

#[tokio::test]
async fn index_report_accounts_for_new_skipped_and_removed() {
    let (idx, _tmp) = fresh_index().await;
    let corpus = corpus_with(&[
        ("a.md", "# A\n\nstable alpha\n"),
        ("b.md", "# B\n\noriginal beta\n"),
        ("c.md", "# C\n\ngamma\n"),
    ]);

    let first = run_index(&idx, corpus.path()).await.unwrap();
    assert_eq!(
        (first.files, first.skipped, first.removed),
        (3, 0, 0),
        "first index: all three files are new"
    );

    // a unchanged, b edited, c removed, d added.
    std::fs::write(corpus.path().join("b.md"), "# B\n\nedited beta\n").unwrap();
    std::fs::remove_file(corpus.path().join("c.md")).unwrap();
    std::fs::write(corpus.path().join("d.md"), "# D\n\ndelta\n").unwrap();

    let second = run_index(&idx, corpus.path()).await.unwrap();
    assert_eq!(
        (second.files, second.skipped, second.removed),
        (2, 1, 1),
        "second index: b+d (re)indexed, a skipped, c pruned"
    );
}

#[tokio::test]
async fn reindex_subdir_does_not_prune_files_outside_it() {
    let (idx, _tmp) = fresh_index().await;
    let corpus = corpus_with(&[
        ("sub1/x.md", "# X\n\nsubonex content\n"),
        ("sub2/y.md", "# Y\n\nsubtwoy content\n"),
    ]);

    // Index the whole corpus, then re-index only one subdirectory.
    run_index(&idx, corpus.path()).await.unwrap();
    run_index(&idx, &corpus.path().join("sub1")).await.unwrap();

    // Pruning must be scoped to the indexed root: sub2's file is outside it
    // and must survive.
    assert!(
        !idx.bm25("subtwoy", 10, None).await.unwrap().is_empty(),
        "indexing sub1 must not prune files under sub2"
    );
    assert!(
        !idx.bm25("subonex", 10, None).await.unwrap().is_empty(),
        "the re-indexed subdir's own file must remain"
    );
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
    // Doubling a term frequency in a doc raises that doc's score (or
    // holds — never drops). We construct a 2-doc corpus where
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
