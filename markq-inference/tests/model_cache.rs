//! Model cache behavior against a local stand-in for the hub.

mod support;

use hf_hub::HFClient;
use support::FakeHub;

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
