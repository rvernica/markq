//! `markq models pull` / `markq models ls` against a local stand-in for the
//! hub, so these tests never touch the network.

mod support;

use markq_cli::models::{ls_rows, run_pull};
use markq_inference::model_cache::{DigestSidecar, KnownModel, ModelRole};
use markq_inference::HFClient;
use support::{Behavior, FakeHub};

fn sample_content() -> Vec<u8> {
    (0..4096_u32).map(|byte| (byte % 251) as u8).collect()
}

#[tokio::test]
async fn pull_unknown_role_lists_valid_roles() {
    let dir = tempfile::tempdir().unwrap();
    // Never contacted: an unknown role name is rejected before any client use.
    let client = HFClient::builder()
        .endpoint("http://127.0.0.1:1")
        .build()
        .unwrap();

    let err = run_pull(Some("embeder"), dir.path(), &client)
        .await
        .unwrap_err();
    let message = err.to_string();
    assert!(message.contains("embeder"), "got: {message}");
    assert!(message.contains("reranker"), "got: {message}");
}

#[tokio::test]
async fn pull_already_verified_transfers_nothing() {
    let model = KnownModel::Qwen3Reranker06B;
    let content = sample_content();
    let hub = FakeHub::spawn_for(model.filename(), content.clone(), Behavior::Complete).await;
    let client = HFClient::builder()
        .endpoint(hub.endpoint())
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();

    let target = dir.path().join(model.filename());
    std::fs::write(&target, &content).unwrap();
    DigestSidecar {
        digest: hub.sha256().to_string(),
        size_bytes: content.len() as u64,
        source: "hub lfs metadata".into(),
    }
    .write_for(&target)
    .unwrap();

    let outcomes = run_pull(Some("reranker"), dir.path(), &client)
        .await
        .unwrap();

    assert_eq!(outcomes.len(), 1);
    assert!(!outcomes[0].transferred);
    assert!(outcomes[0].is_ready());
    assert_eq!(
        hub.request_count(),
        0,
        "an already-verified model must not contact the hub"
    );
}

#[tokio::test]
async fn pull_all_roles_reports_each_ready() {
    let embedder = KnownModel::Qwen3Embedding06B;
    let reranker = KnownModel::Qwen3Reranker06B;
    let dir = tempfile::tempdir().unwrap();

    // One hub, reused for both roles: each role's request names its own
    // filename, and the fake hub answers metadata/content/tree requests for
    // whatever filename the caller asks about.
    let hub = FakeHub::spawn_for(embedder.filename(), sample_content(), Behavior::Complete).await;
    let client = HFClient::builder()
        .endpoint(hub.endpoint())
        .build()
        .unwrap();

    // The reranker's file lives under a different name than the hub
    // advertises, so pre-stage it as already verified to keep this test
    // about the "every role" fan-out rather than a second hub.
    let reranker_target = dir.path().join(reranker.filename());
    let reranker_content = sample_content();
    std::fs::write(&reranker_target, &reranker_content).unwrap();
    let reranker_digest = markq_inference::sha256_hex(&reranker_target).unwrap();
    DigestSidecar {
        digest: reranker_digest,
        size_bytes: reranker_content.len() as u64,
        source: "hub lfs metadata".into(),
    }
    .write_for(&reranker_target)
    .unwrap();

    let outcomes = run_pull(None, dir.path(), &client).await.unwrap();

    assert_eq!(outcomes.len(), ModelRole::ALL.len());
    for outcome in &outcomes {
        assert!(
            outcome.is_ready(),
            "role {} did not report ready: {:?}",
            outcome.role.as_str(),
            outcome.result
        );
    }
}

#[tokio::test]
async fn pull_then_ls_then_repull_is_a_noop() {
    let model = KnownModel::Qwen3Embedding06B;
    let content = sample_content();
    let hub = FakeHub::spawn_for(model.filename(), content.clone(), Behavior::Complete).await;
    let client = HFClient::builder()
        .endpoint(hub.endpoint())
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();

    // An empty cache: pulling the embedder fetches it from scratch.
    let first_pull = run_pull(Some("embedder"), dir.path(), &client)
        .await
        .unwrap();
    assert_eq!(first_pull.len(), 1);
    assert!(first_pull[0].transferred);
    assert!(first_pull[0].is_ready());
    let requests_after_first_pull = hub.request_count();
    assert!(
        requests_after_first_pull > 0,
        "an empty cache must contact the hub to fetch the model"
    );

    // `ls` reports it verified, with the hub's declared digest and size.
    let rows = ls_rows(dir.path()).unwrap();
    let embedder_row = rows
        .iter()
        .find(|row| row.role == "embedder")
        .expect("embedder row present");
    assert_eq!(embedder_row.state, "verified");
    assert_eq!(embedder_row.size_bytes, Some(content.len() as u64));
    assert_eq!(embedder_row.digest.as_deref(), Some(hub.sha256()));

    // A second pull transfers nothing and contacts the hub no further.
    let second_pull = run_pull(Some("embedder"), dir.path(), &client)
        .await
        .unwrap();
    assert_eq!(second_pull.len(), 1);
    assert!(!second_pull[0].transferred);
    assert!(second_pull[0].is_ready());
    assert_eq!(
        hub.request_count(),
        requests_after_first_pull,
        "re-pulling an already-verified model must not contact the hub again"
    );
}
