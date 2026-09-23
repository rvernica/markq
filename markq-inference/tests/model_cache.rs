//! Model cache behavior against a local stand-in for the hub.

use fs4::FileExt;
use hf_hub::HFClient;
use markq_inference::model_cache::{
    declared_digest, download_model, lock_path_for, partial_path_for, CacheEntry, CacheState,
    KnownModel, ModelRole,
};
use markq_test_support::{Behavior, FakeHub};

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

    let path = download_model(&client, model, dir.path(), None)
        .await
        .unwrap();

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

    let result = download_model(&client, model, dir.path(), None).await;

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

#[tokio::test]
async fn second_download_of_same_model_fails_immediately() {
    let model = KnownModel::Qwen3Embedding06B;
    let hub = FakeHub::spawn_for(model.filename(), sample_content(), Behavior::Complete).await;
    let client = HFClient::builder()
        .endpoint(hub.endpoint())
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path()).unwrap();
    let target = dir.path().join(model.filename());

    // Stand in for another process already downloading this model.
    let held = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lock_path_for(&target))
        .unwrap();
    held.try_lock_exclusive().unwrap();

    let result = download_model(&client, model, dir.path(), None).await;

    let message = result.unwrap_err().to_string();
    assert!(
        message.contains("already running"),
        "expected an in-progress message, got: {message}"
    );
    assert_eq!(
        hub.request_count(),
        0,
        "a contended lock must be refused before the hub is contacted"
    );
}

#[tokio::test]
async fn downloads_of_different_models_do_not_block() {
    let embedder = KnownModel::Qwen3Embedding06B;
    let reranker = KnownModel::Qwen3Reranker06B;
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path()).unwrap();
    let embedder_target = dir.path().join(embedder.filename());

    // Hold the embedder's lock, as if its download were already in progress.
    let held = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lock_path_for(&embedder_target))
        .unwrap();
    held.try_lock_exclusive().unwrap();

    let content = sample_content();
    let hub = FakeHub::spawn_for(reranker.filename(), content, Behavior::Complete).await;
    let client = HFClient::builder()
        .endpoint(hub.endpoint())
        .build()
        .unwrap();

    let result = download_model(&client, reranker, dir.path(), None).await;

    assert!(
        result.is_ok(),
        "a lock on a different model must not block this one: {:?}",
        result.err()
    );
}

#[tokio::test]
async fn failed_verification_leaves_no_loadable_model() {
    let model = KnownModel::Qwen3Embedding06B;
    let hub = FakeHub::spawn_for(
        model.filename(),
        sample_content(),
        Behavior::WrongDeclaredDigest,
    )
    .await;
    let client = HFClient::builder()
        .endpoint(hub.endpoint())
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join(model.filename());

    let error = download_model(&client, model, dir.path(), None)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("digest"),
        "expected a digest complaint, got: {error}"
    );

    // A file markq knows failed its own digest check must not be left where a
    // later load would mistake it for an operator-staged file.
    assert!(
        !target.exists(),
        "a file that failed verification is sitting at the model's own path"
    );
    assert_eq!(
        CacheEntry::inspect(ModelRole::Embedder, dir.path())
            .unwrap()
            .state,
        CacheState::Partial { bytes: 4096 }
    );
}

#[tokio::test]
async fn offline_missing_model_names_model_and_setting() {
    let model = KnownModel::Qwen3Embedding06B;
    let hub = FakeHub::spawn_for(model.filename(), sample_content(), Behavior::Complete).await;
    let client = HFClient::builder()
        .endpoint(hub.endpoint())
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();

    let error = download_model(&client, model, dir.path(), Some("MARKQ_OFFLINE"))
        .await
        .unwrap_err();
    let message = error.to_string();

    assert!(message.contains("embedder"), "got: {message}");
    assert!(message.contains("MARKQ_OFFLINE"), "got: {message}");
}

#[tokio::test]
async fn offline_pull_refuses_without_contacting_the_hub() {
    let model = KnownModel::Qwen3Embedding06B;
    let hub = FakeHub::spawn_for(model.filename(), sample_content(), Behavior::Complete).await;
    let client = HFClient::builder()
        .endpoint(hub.endpoint())
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();

    let _ = download_model(&client, model, dir.path(), Some("--offline")).await;

    assert_eq!(
        hub.request_count(),
        0,
        "offline mode must refuse before reaching the hub"
    );
}

#[tokio::test]
async fn complete_cache_makes_no_request_with_offline_unset() {
    let model = KnownModel::Qwen3Embedding06B;
    let content = sample_content();
    let hub = FakeHub::spawn_for(model.filename(), content.clone(), Behavior::Complete).await;
    let client = HFClient::builder()
        .endpoint(hub.endpoint())
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();

    // Fill the cache the way a completed download leaves it.
    download_model(&client, model, dir.path(), None)
        .await
        .unwrap();
    let after_fetch = hub.request_count();
    assert!(after_fetch > 0, "the first fetch should have used the hub");

    // A cache that is already complete needs nothing, offline flag or not.
    download_model(&client, model, dir.path(), None)
        .await
        .unwrap();
    assert_eq!(
        hub.request_count(),
        after_fetch,
        "a verified cache must not produce another request"
    );
}
