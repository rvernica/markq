//! `markq doctor` end to end: structured JSON output, the no-network and
//! no-weights guarantees (FR-020, FR-022), read-only behavior (FR-025), and
//! roadmap-language-free messages (FR-027).

use markq_cli::doctor::run_doctor;

fn snapshot_dir(dir: &std::path::Path) -> Vec<(String, Vec<u8>)> {
    if !dir.exists() {
        return Vec::new();
    }
    let mut entries: Vec<(String, Vec<u8>)> = walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| {
            let relative = entry
                .path()
                .strip_prefix(dir)
                .unwrap()
                .to_string_lossy()
                .to_string();
            let content = std::fs::read(entry.path()).unwrap();
            (relative, content)
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

#[tokio::test]
async fn doctor_json_emits_check_severity_observed_expected_next_step() {
    let dataset_dir = tempfile::tempdir().unwrap();
    let dataset_path = dataset_dir.path().join("chunks.lance");
    let models_dir = tempfile::tempdir().unwrap();

    let findings = run_doctor(&dataset_path, models_dir.path(), false)
        .await
        .unwrap();

    let value = serde_json::to_value(&findings).unwrap();
    let array = value.as_array().unwrap();
    assert!(!array.is_empty());
    for entry in array {
        let object = entry.as_object().unwrap();
        for key in ["check", "severity", "observed", "expected", "next_step"] {
            assert!(object.contains_key(key), "finding missing {key:?}: {entry}");
        }
    }
}

/// Doctor must never load model weights (FR-020), which is exactly what may
/// be broken on the machine it's diagnosing. Both the embedder and reranker
/// loaders route through `markq_inference::backend::shared()`, which logs
/// "LlamaBackend initialized" the first time (and only the first time) the
/// llama.cpp backend actually spins up — so its absence from a captured log
/// proves no model load was attempted.
#[tokio::test(flavor = "current_thread")]
async fn doctor_loads_no_model_weights() {
    #[derive(Clone, Default)]
    struct CapturedLogs(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for CapturedLogs {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLogs {
        type Writer = CapturedLogs;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    let captured = CapturedLogs::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(captured.clone())
        .with_max_level(tracing::Level::DEBUG)
        .finish();
    let dispatch = tracing::Dispatch::new(subscriber);

    let dataset_dir = tempfile::tempdir().unwrap();
    let dataset_path = dataset_dir.path().join("chunks.lance");
    let models_dir = tempfile::tempdir().unwrap();

    let findings = {
        let _guard = tracing::dispatcher::set_default(&dispatch);
        run_doctor(&dataset_path, models_dir.path(), false)
            .await
            .unwrap()
    };
    assert!(!findings.is_empty());

    let text = String::from_utf8(captured.0.lock().unwrap().clone()).unwrap();
    assert!(
        !text.contains("LlamaBackend initialized"),
        "doctor must not load model weights, but the backend was initialized: {text}"
    );
}

#[tokio::test]
async fn doctor_runs_with_network_forbidden() {
    // Point any accidental network use at an address nothing answers, so a
    // regression that makes doctor contact the network fails loudly instead
    // of silently succeeding through the real hub.
    std::env::set_var("HTTPS_PROXY", "http://127.0.0.1:1");
    std::env::set_var("HTTP_PROXY", "http://127.0.0.1:1");

    let dataset_dir = tempfile::tempdir().unwrap();
    let dataset_path = dataset_dir.path().join("chunks.lance");
    let models_dir = tempfile::tempdir().unwrap();

    let findings = run_doctor(&dataset_path, models_dir.path(), false)
        .await
        .unwrap();

    std::env::remove_var("HTTPS_PROXY");
    std::env::remove_var("HTTP_PROXY");

    assert!(
        !findings.is_empty(),
        "doctor must still produce the full finding set with network forbidden"
    );
}

#[tokio::test]
async fn doctor_mutates_nothing() {
    let dataset_dir = tempfile::tempdir().unwrap();
    let dataset_path = dataset_dir.path().join("chunks.lance");
    markq_index_lance::LanceIndex::open_or_create(&dataset_path)
        .await
        .unwrap();

    let models_dir = tempfile::tempdir().unwrap();
    let role = markq_inference::model_cache::ModelRole::Embedder;
    let model_path = models_dir.path().join(role.known_model().filename());
    let content: Vec<u8> = (0..2048_u32).map(|byte| (byte % 251) as u8).collect();
    std::fs::write(&model_path, &content).unwrap();
    markq_inference::model_cache::DigestSidecar {
        digest: markq_inference::sha256_hex(&model_path).unwrap(),
        size_bytes: content.len() as u64,
        source: "hub lfs metadata".into(),
    }
    .write_for(&model_path)
    .unwrap();

    let before_dataset = snapshot_dir(&dataset_path);
    let before_models = snapshot_dir(models_dir.path());

    run_doctor(&dataset_path, models_dir.path(), false)
        .await
        .unwrap();
    assert_eq!(
        before_dataset,
        snapshot_dir(&dataset_path),
        "a default doctor run must not mutate the dataset"
    );
    assert_eq!(
        before_models,
        snapshot_dir(models_dir.path()),
        "a default doctor run must not mutate the model cache"
    );

    run_doctor(&dataset_path, models_dir.path(), true)
        .await
        .unwrap();
    assert_eq!(
        before_dataset,
        snapshot_dir(&dataset_path),
        "a --verify doctor run must not mutate the dataset"
    );
    assert_eq!(
        before_models,
        snapshot_dir(models_dir.path()),
        "a --verify doctor run must not mutate the model cache"
    );
}

#[tokio::test]
async fn diagnostic_messages_are_free_of_roadmap_language() {
    let dataset_dir = tempfile::tempdir().unwrap();
    let dataset_path = dataset_dir.path().join("chunks.lance");
    let models_dir = tempfile::tempdir().unwrap();

    let findings = run_doctor(&dataset_path, models_dir.path(), false)
        .await
        .unwrap();
    assert!(!findings.is_empty());

    for finding in &findings {
        let fields = [
            Some(finding.check.clone()),
            finding.observed.clone(),
            finding.expected.clone(),
            Some(finding.next_step.clone()),
        ];
        for text in fields.into_iter().flatten() {
            let lower = text.to_ascii_lowercase();
            assert!(!lower.contains("phase"), "roadmap language in {text:?}");
            assert!(!lower.contains("qmd/"), "qmd citation in {text:?}");
        }
    }
}
