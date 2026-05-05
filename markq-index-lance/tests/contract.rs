//! Run the markq-core `Index` contract against the Lance backend.

use markq_core::contract::run_contract;
use markq_index_lance::LanceIndex;

#[tokio::test]
async fn lance_satisfies_index_contract() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chunks.lance");
    let idx = LanceIndex::open_or_create(&path).await.unwrap();
    run_contract(&idx).await;
}
