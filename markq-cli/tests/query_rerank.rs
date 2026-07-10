//! Integration test for `run_query`'s post-fusion `--rerank` step: index +
//! embed a small corpus crafted so BM25+vector fusion and the cross-encoder
//! rerank DISAGREE on ordering, then assert:
//!   - `rerank=false` reproduces the plain fusion order (untouched path).
//!   - `rerank=true` reorders the fused pool, promoting the genuinely
//!     on-topic chunk that fusion under-ranked.
//!   - top-k is honored in both modes.
//!
//! Gated on BOTH `$MARKQ_TEST_MODEL` (embedder, needed by `query`'s vector
//! half) and `$MARKQ_TEST_RERANK_MODEL` (cross-encoder reranker). Both model
//! files live under the same `MARKQ_MODELS_DIR` in practice, e.g.
//! `~/.cache/markq/models/{Qwen3-Embedding-0.6B-Q8_0.gguf,qwen3-reranker-0.6b-q8_0.gguf}`.
//! Run with:
//!
//!   MARKQ_MODELS_DIR=/home/vernica/.cache/markq/models \
//!   MARKQ_TEST_MODEL=/home/vernica/.cache/markq/models/Qwen3-Embedding-0.6B-Q8_0.gguf \
//!   MARKQ_TEST_RERANK_MODEL=/home/vernica/.cache/markq/models/qwen3-reranker-0.6b-q8_0.gguf \
//!     cargo test -p markq-cli --test query_rerank -- --ignored --nocapture

use std::fs;
use std::path::PathBuf;

use markq_cli::{indexer, query, search};
use markq_index_lance::LanceIndex;
use markq_inference::KnownModel;
use tempfile::TempDir;

/// Resolve `$MARKQ_TEST_MODEL` (embedder) and `$MARKQ_TEST_RERANK_MODEL`
/// (reranker), assert each points at the exact filename `ensure_model`
/// expects, and wire `MARKQ_MODELS_DIR` to their shared parent so both
/// `ensure_model` calls resolve locally instead of hitting the network.
/// Returns `None` (so the caller can early-return) unless both are set.
fn setup_test_models() -> Option<()> {
    let embed_path = PathBuf::from(std::env::var("MARKQ_TEST_MODEL").ok()?);
    let rerank_path = PathBuf::from(std::env::var("MARKQ_TEST_RERANK_MODEL").ok()?);

    assert!(
        embed_path.exists(),
        "MARKQ_TEST_MODEL={} does not exist",
        embed_path.display()
    );
    assert!(
        rerank_path.exists(),
        "MARKQ_TEST_RERANK_MODEL={} does not exist",
        rerank_path.display()
    );

    let expected_embed_filename = KnownModel::Qwen3Embedding06B.filename();
    assert_eq!(
        embed_path.file_name().and_then(|s| s.to_str()),
        Some(expected_embed_filename),
        "MARKQ_TEST_MODEL must point at a file literally named {expected_embed_filename}",
    );
    let expected_rerank_filename = KnownModel::Qwen3Reranker06B.filename();
    assert_eq!(
        rerank_path.file_name().and_then(|s| s.to_str()),
        Some(expected_rerank_filename),
        "MARKQ_TEST_RERANK_MODEL must point at a file literally named {expected_rerank_filename}",
    );

    let embed_parent = embed_path
        .parent()
        .expect("MARKQ_TEST_MODEL must have a parent");
    let rerank_parent = rerank_path
        .parent()
        .expect("MARKQ_TEST_RERANK_MODEL must have a parent");
    assert_eq!(
        embed_parent, rerank_parent,
        "this test wires a single MARKQ_MODELS_DIR; point both env vars at \
         files in the same directory (e.g. ~/.cache/markq/models)",
    );
    std::env::set_var("MARKQ_MODELS_DIR", embed_parent);
    Some(())
}

/// Corpus engineered so BM25+vector fusion and the cross-encoder rerank
/// disagree:
///   - `keyword_stuffed.md` repeats the query's literal keywords but in a
///     nonsensical, off-topic sentence — strong lexical (BM25) signal, weak
///     real relevance.
///   - `on_topic.md` genuinely answers the query but in paraphrase, sharing
///     almost no literal keywords with it — weak BM25 signal, but the
///     semantically correct answer, which the cross-encoder should surface.
///   - `off_topic.md` is unrelated filler with neither lexical nor semantic
///     overlap, to keep the fused/vector candidate pool from being trivial.
fn write_corpus(dir: &std::path::Path) {
    fs::write(
        dir.join("keyword_stuffed.md"),
        "# Quarterly Newsletter\n\n\
         Sunlight Energy Corp's stock did not convert investor confidence \
         this quarter. Our internal codename for the marketing campaign \
         will convert customer sunlight-hour credits into energy rebates. \
         Please review the attached invoice for sunlight energy usage and \
         convert your rebate before the March deadline.\n",
    )
    .unwrap();
    fs::write(
        dir.join("on_topic.md"),
        "# How Plants Make Food\n\n\
         Photosynthesis is how a leaf's chlorophyll captures light from the \
         sun and drives a chemical reaction that turns water and carbon \
         dioxide into glucose, the fuel a plant uses to grow.\n",
    )
    .unwrap();
    fs::write(
        dir.join("off_topic.md"),
        "# Sourdough Timing\n\n\
         Feed the starter twelve hours before mixing the dough, then let the \
         bulk ferment run for four hours at room temperature.\n",
    )
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MARKQ_TEST_MODEL and MARKQ_TEST_RERANK_MODEL"]
async fn rerank_reorders_the_fused_pool_after_fusion() {
    if setup_test_models().is_none() {
        return; // belt-and-braces; the #[ignore] above already gates this
    }

    let dir = TempDir::new().expect("tempdir");
    let corpus = dir.path().join("corpus");
    fs::create_dir_all(&corpus).unwrap();
    write_corpus(&corpus);

    let dataset = dir.path().join("chunks.lance");
    let idx = LanceIndex::open_or_create(&dataset)
        .await
        .expect("open dataset");
    let index_report = indexer::run_index(&idx, &corpus).await.expect("index");
    assert_eq!(index_report.chunks, 3);

    let embed_report = markq_cli::embedder_cmd::run_embed(&idx)
        .await
        .expect("embed");
    assert_eq!(embed_report.rows, 3);

    // Reopen: `run_index`'s FTS index and `run_embed`'s vector index are not
    // visible to a handle opened before either ran.
    let idx = LanceIndex::open(&dataset).await.expect("reopen");

    let query_text = "How does photosynthesis convert sunlight into energy?";
    let opts = search::SearchOptions {
        top_k: Some(3),
        min_score: None,
    };

    let fusion_only = query::run_query(&idx, query_text, 3, &opts, false, false)
        .await
        .expect("run_query without rerank");
    assert!(
        !fusion_only.hits.is_empty(),
        "expected at least one fused hit"
    );
    assert!(
        fusion_only.explain.is_none(),
        "explain must stay off when not requested"
    );

    // The keyword-stuffed doc's heavy literal term overlap should win fusion.
    let fused_order: Vec<String> = fusion_only.hits.iter().map(|h| h.path.clone()).collect();
    assert!(
        fused_order[0].ends_with("keyword_stuffed.md"),
        "expected fusion to rank the keyword-stuffed doc first, got {fused_order:?}",
    );

    let reranked = query::run_query(&idx, query_text, 3, &opts, false, true)
        .await
        .expect("run_query with rerank");
    let rerank_order: Vec<String> = reranked.hits.iter().map(|h| h.path.clone()).collect();
    eprintln!("fusion order: {fused_order:?}");
    eprintln!(
        "rerank order: {rerank_order:?} (scores: {:?})",
        reranked
            .hits
            .iter()
            .map(|h| (h.path.clone(), h.score))
            .collect::<Vec<_>>()
    );

    // Reranking must actually change the order versus fusion, promoting the
    // genuinely on-topic (paraphrased) chunk to the top.
    assert_ne!(
        rerank_order, fused_order,
        "rerank should reorder the fused pool, not just replay fusion order"
    );
    assert!(
        rerank_order[0].ends_with("on_topic.md"),
        "expected rerank to promote the on-topic doc to rank 1, got {rerank_order:?}",
    );

    // top-k is honored post-rerank.
    let top1 = query::run_query(
        &idx,
        query_text,
        3,
        &search::SearchOptions {
            top_k: Some(1),
            min_score: None,
        },
        false,
        true,
    )
    .await
    .expect("run_query with rerank + top_k=1");
    assert_eq!(top1.hits.len(), 1, "top_k=1 must be honored after rerank");
    assert!(top1.hits[0].path.ends_with("on_topic.md"));
}
