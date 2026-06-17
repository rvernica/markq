//! Schema round-trip: build a chunk RecordBatch, upsert, reopen, assert the
//! dataset reports the same field shape and the row is there.

use std::sync::Arc;

use arrow_array::{
    builder::{FixedSizeListBuilder, Float32Builder},
    Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use markq_core::{chunk_arrow_schema, ChunkColumn, Index, EMBEDDING_DIM_DEFAULT, SCHEMA_VERSION};
use markq_index_lance::LanceIndex;

fn null_embedding(rows: usize, dim: i32) -> Arc<dyn Array> {
    // FixedSizeList where every row's *list* is null. Builder requires
    // appending a value before nulling, so we append zeros + mark null.
    let values = Float32Builder::with_capacity(rows * dim as usize);
    let mut b = FixedSizeListBuilder::new(values, dim);
    for _ in 0..rows {
        for _ in 0..dim {
            b.values().append_value(0.0);
        }
        b.append(false);
    }
    Arc::new(b.finish())
}

#[tokio::test]
async fn chunk_record_batch_round_trips_through_lance() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chunks.lance");
    let idx = LanceIndex::open_or_create(&path).await.unwrap();
    let schema = chunk_arrow_schema(EMBEDDING_DIM_DEFAULT);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["chunk-1"])),
            Arc::new(StringArray::from(vec!["default"])),
            Arc::new(StringArray::from(vec!["markq://default/notes/a.md"])),
            Arc::new(StringArray::from(vec!["notes/a.md"])),
            Arc::new(StringArray::from(vec!["deadbeef"])),
            Arc::new(Int64Array::from(vec![1_700_000_000_000_000_000])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(StringArray::from(vec!["hello world"])),
            // tokens, embedding, context_id are nullable
            Arc::new(Int32Array::from(vec![Option::<i32>::None])),
            null_embedding(1, EMBEDDING_DIM_DEFAULT),
            Arc::new(StringArray::from(vec![Option::<&str>::None])),
            Arc::new(Int32Array::from(vec![SCHEMA_VERSION as i32])),
        ],
    )
    .unwrap();

    idx.upsert_chunks(vec![batch]).await.unwrap();
    assert_eq!(idx.count_rows().await.unwrap(), 1);

    // Reopen: schema and row count survive.
    drop(idx);
    let idx2 = LanceIndex::open_or_create(&path).await.unwrap();
    assert_eq!(idx2.count_rows().await.unwrap(), 1);
    let s2 = idx2.arrow_schema();
    let names: Vec<&str> = s2.fields().iter().map(|f| f.name().as_str()).collect();
    assert!(names.contains(&ChunkColumn::EMBEDDING));
    assert!(names.contains(&ChunkColumn::CONTEXT_ID));
}
