//! Edge-case coverage for `run_rerank` that must be testable in CI WITHOUT a
//! reranker model: empty input short-circuits before model load, and
//! malformed candidates (missing/empty `id`/`text`) are rejected during
//! validation, which also happens before model load.

use markq_cli::rerank::run_rerank;

#[tokio::test]
async fn empty_array_short_circuits_before_model_load_json() {
    // No MARKQ_MODELS_DIR / network access is configured in this test, so if
    // `run_rerank` tried to load the model it would fail. Getting `Ok(())`
    // back proves the empty short-circuit happens before model load.
    let mut buf = Vec::new();
    run_rerank(b"[]".as_slice(), &mut buf, "q", None, None, true)
        .await
        .expect("empty input must short-circuit to Ok without loading a model");

    let out = String::from_utf8(buf).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(out.trim()).expect("valid JSON array");
    assert_eq!(parsed.as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn empty_array_short_circuits_before_model_load_human() {
    let mut buf = Vec::new();
    run_rerank(b"[]".as_slice(), &mut buf, "q", None, None, false)
        .await
        .expect("empty input must short-circuit to Ok without loading a model");

    // Human output for empty input is truly empty (not "(no results)").
    assert_eq!(buf, Vec::<u8>::new());
}

#[tokio::test]
async fn missing_text_field_names_the_offending_index() {
    let mut buf = Vec::new();
    let err = run_rerank(
        br#"[{"id":"a"}]"#.as_slice(),
        &mut buf,
        "q",
        None,
        None,
        true,
    )
    .await
    .expect_err("missing text must error before model load");

    let msg = format!("{err:#}");
    assert!(msg.contains("index 0"), "message: {msg}");
    assert!(msg.contains("text"), "message: {msg}");
}

#[tokio::test]
async fn missing_id_field_names_the_offending_index() {
    let mut buf = Vec::new();
    let err = run_rerank(
        br#"[{"text":"x"}]"#.as_slice(),
        &mut buf,
        "q",
        None,
        None,
        true,
    )
    .await
    .expect_err("missing id must error before model load");

    let msg = format!("{err:#}");
    assert!(msg.contains("index 0"), "message: {msg}");
    assert!(msg.contains("id"), "message: {msg}");
}

#[tokio::test]
async fn empty_id_field_names_the_offending_index() {
    let mut buf = Vec::new();
    let err = run_rerank(
        br#"[{"id":"","text":"x"}]"#.as_slice(),
        &mut buf,
        "q",
        None,
        None,
        true,
    )
    .await
    .expect_err("empty id must error before model load");

    let msg = format!("{err:#}");
    assert!(msg.contains("index 0"), "message: {msg}");
    assert!(msg.contains("id"), "message: {msg}");
}

#[tokio::test]
async fn second_element_names_index_one() {
    let mut buf = Vec::new();
    let err = run_rerank(
        br#"[{"id":"a","text":"ok"},{"id":"b"}]"#.as_slice(),
        &mut buf,
        "q",
        None,
        None,
        true,
    )
    .await
    .expect_err("missing text on second element must name index 1");

    let msg = format!("{err:#}");
    assert!(msg.contains("index 1"), "message: {msg}");
    assert!(msg.contains("text"), "message: {msg}");
}
