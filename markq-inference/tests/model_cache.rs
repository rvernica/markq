//! Model cache behavior against a local stand-in for the hub.

mod support;

use hf_hub::HFClient;
use markq_inference::model_cache::{
    declared_digest, download_model, partial_path_for, CacheEntry, CacheState, KnownModel,
    ModelRole,
};
use support::{Behavior, FakeHub};

fn sample_content() -> Vec<u8> {
    (0..4096_u32).map(|byte| (byte % 251) as u8).collect()
}

#[tokio::test]
async fn serves_full_file_and_honors_range() {
    let hub = FakeHub::spawn(sample_content()).await;
    let client = HFClient::builder()
        .endpoint(hub.endpoint())
        .build()
        .unwrap();
    let repo = client.model("markq-test", "models");

    let full = repo
        .download_file_to_bytes()
        .filename("model.gguf")
        .send()
        .await
        .unwrap();
    assert_eq!(full.as_ref(), hub.content());

    let tail = repo
        .download_file_to_bytes()
        .filename("model.gguf")
        .range(1024..2048)
        .send()
        .await
        .unwrap();
    assert_eq!(tail.as_ref(), &hub.content()[1024..2048]);
}

#[tokio::test]
async fn declared_digest_read_from_hub_metadata() {
    let model = KnownModel::Qwen3Embedding06B;
    let hub = FakeHub::spawn_for(model.filename(), sample_content(), Behavior::Complete).await;
    let client = HFClient::builder()
        .endpoint(hub.endpoint())
        .build()
        .unwrap();

    let declared = declared_digest(&client, model).await.unwrap();
    assert_eq!(declared.as_deref(), Some(hub.sha256()));
}

#[tokio::test]
async fn absent_declared_digest_yields_unverified() {
    let model = KnownModel::Qwen3Embedding06B;
    let hub = FakeHub::spawn_for(
        model.filename(),
        sample_content(),
        Behavior::NoDeclaredDigest,
    )
    .await;
    let client = HFClient::builder()
        .endpoint(hub.endpoint())
        .build()
        .unwrap();

    assert_eq!(declared_digest(&client, model).await.unwrap(), None);
}

#[tokio::test]
async fn interrupted_download_resumes_from_partial() {
    let model = KnownModel::Qwen3Embedding06B;
    let content = sample_content();
    let hub = FakeHub::spawn_for(model.filename(), content.clone(), Behavior::Complete).await;
    let client = HFClient::builder()
        .endpoint(hub.endpoint())
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();

    // Stand in for a transfer that stopped a quarter of the way through.
    let target = dir.path().join(model.filename());
    let already_have = 1024;
    std::fs::write(partial_path_for(&target), &content[..already_have]).unwrap();

    let path = download_model(&client, model, dir.path()).await.unwrap();

    assert_eq!(std::fs::read(&path).unwrap(), content);
    assert_eq!(hub.last_range().map(|(start, _)| start), Some(1024));
    assert!(
        !partial_path_for(&target).exists(),
        "partial file left behind"
    );
}

#[tokio::test]
async fn partial_file_is_never_used_as_complete() {
    let model = KnownModel::Qwen3Embedding06B;
    let content = sample_content();
    let hub = FakeHub::spawn_for(
        model.filename(),
        content.clone(),
        Behavior::TruncateAfter(512),
    )
    .await;
    let client = HFClient::builder()
        .endpoint(hub.endpoint())
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join(model.filename());

    let result = download_model(&client, model, dir.path()).await;

    assert!(result.is_err(), "a short transfer must not report success");
    assert!(
        !target.exists(),
        "an incomplete transfer must not land as the model file"
    );
    assert_eq!(
        CacheEntry::inspect(ModelRole::Embedder, dir.path())
            .unwrap()
            .state,
        CacheState::Partial { bytes: 512 }
    );
}
