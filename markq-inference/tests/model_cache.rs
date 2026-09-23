//! Model cache behavior against a local stand-in for the hub.

mod support;

use hf_hub::HFClient;
use markq_inference::model_cache::{declared_digest, KnownModel};
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
