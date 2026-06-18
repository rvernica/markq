//! Backend contract tests for the `Index` trait. `markq-index-lance` calls
//! into this; the suite is structured so a second backend (if
//! LanceDB churn ever forces one) can run the same tests without forking.
//!
//! Gated by the `test-harness` feature so the contract doesn't bloat
//! release binaries.

use arrow_schema::DataType;

use crate::index::Index;
use crate::metadata::SCHEMA_VERSION;
use crate::schema::ChunkColumn;

/// Run the full contract suite against a freshly-opened backend.
/// Caller is responsible for tearing down the underlying storage.
pub async fn run_contract<I: Index>(idx: &I) {
    // Schema is the canonical v1 shape.
    let schema = idx.arrow_schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names.first().copied(), Some("id"));
    assert!(names.contains(&"embedding"));
    assert!(names.contains(&"context_id"));
    assert!(names.contains(&"schema_version"));

    // Empty backend.
    assert_eq!(idx.count_rows().await.expect("count_rows"), 0);

    // Metadata records the v1 schema version and a non-empty Lance format
    // string. (PHASE1_FOLLOWUPS #4.)
    let md = idx.metadata().await.expect("metadata");
    assert_eq!(md.schema_version, SCHEMA_VERSION);
    assert!(!md.lance_file_format_version.is_empty());
    assert!(!md.lancedb_crate_version.is_empty());

    // Query methods are wired but return empty on an empty backend. The
    // contract is "no panic, returns an empty Vec" so the real retrieval
    // paths can light them up without changing trait shape.
    assert!(idx
        .bm25("anything", 10, None)
        .await
        .expect("bm25")
        .is_empty());
    // Size the dummy embedding from the schema rather than hardcoding it —
    // once the real KNN path is wired, a mis-sized vector would start
    // failing here exactly when the contract should be lighting up.
    let embedding_field = schema
        .field_with_name(ChunkColumn::EMBEDDING)
        .expect("embedding column");
    let embedding_dim = match embedding_field.data_type() {
        DataType::FixedSizeList(_, n) => *n as usize,
        other => panic!("expected FixedSizeList embedding column, got {other:?}"),
    };
    let dummy = vec![0.0f32; embedding_dim];
    assert!(idx
        .vector(&dummy, 10, None)
        .await
        .expect("vector")
        .is_empty());
    let (lex, vec) = idx.hybrid("x", &dummy, 10, None).await.expect("hybrid");
    assert!(lex.is_empty() && vec.is_empty());

    // Empty upsert is a no-op.
    idx.upsert_chunks(Vec::new()).await.expect("empty upsert");

    // Delete on a path with no matching rows is a 0-row no-op.
    let removed = idx
        .delete_by_path("default", "does/not/exist.md")
        .await
        .expect("delete_by_path");
    assert_eq!(removed, 0);

    // existing_file_hashes on an empty backend returns nothing.
    assert!(idx
        .existing_file_hashes("default")
        .await
        .expect("existing_file_hashes")
        .is_empty());

    // Compact on an empty index is a no-op.
    idx.compact().await.expect("compact");
}
