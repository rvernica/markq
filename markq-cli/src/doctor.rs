//! `markq doctor`: one read-only pass that reports the dataset's presence
//! and identity, index coverage, and model cache state, each finding carrying
//! a severity and a single actionable next step.

use std::cmp::Ordering;
use std::path::Path;

use anyhow::Context;
use markq_core::{
    ChunkColumn, DatasetMetadata, Index, IndexHealth, EMBEDDING_DIM_DEFAULT, SCHEMA_VERSION,
};
use markq_index_lance::LanceIndex;
use markq_inference::model_cache::{self, CacheEntry, CacheState, DigestSidecar, ModelRole};
use serde::Serialize;

/// Severity of one diagnostic finding (data-model.md `Diagnostic`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Ok,
    Warning,
    Error,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Severity::Ok => "ok",
            Severity::Warning => "warning",
            Severity::Error => "error",
        })
    }
}

/// One finding from `markq doctor`. `next_step` is required to be non-empty
/// whenever `severity` is not `Ok` (FR-019); an `Ok` finding may still carry
/// one (e.g. "not yet initialized, run `markq index`" is `Ok`, not an error,
/// per FR-023).
#[derive(Debug, Clone, Serialize)]
pub struct Diagnostic {
    pub check: String,
    pub severity: Severity,
    pub observed: Option<String>,
    pub expected: Option<String>,
    pub next_step: String,
}

impl Diagnostic {
    pub fn new(
        check: impl Into<String>,
        severity: Severity,
        observed: Option<String>,
        expected: Option<String>,
        next_step: impl Into<String>,
    ) -> Self {
        Self {
            check: check.into(),
            severity,
            observed,
            expected,
            next_step: next_step.into(),
        }
    }

    /// A finding with nothing wrong and nothing to do.
    pub fn ok(check: impl Into<String>) -> Self {
        Self::new(check, Severity::Ok, None, None, "")
    }

    pub fn warning(
        check: impl Into<String>,
        observed: impl Into<String>,
        expected: impl Into<String>,
        next_step: impl Into<String>,
    ) -> Self {
        Self::new(
            check,
            Severity::Warning,
            Some(observed.into()),
            Some(expected.into()),
            next_step,
        )
    }

    pub fn error(
        check: impl Into<String>,
        observed: impl Into<String>,
        expected: impl Into<String>,
        next_step: impl Into<String>,
    ) -> Self {
        Self::new(
            check,
            Severity::Error,
            Some(observed.into()),
            Some(expected.into()),
            next_step,
        )
    }
}

/// Whether the process should exit with a failure status: true when any
/// finding is `Severity::Error` (FR-021). Warnings alone do not fail.
pub fn any_error(findings: &[Diagnostic]) -> bool {
    findings.iter().any(|f| f.severity == Severity::Error)
}

/// Dataset presence, schema version, and embedder identity (FR-018, FR-023).
/// Never creates a dataset (FR-025): a path that does not exist is reported
/// as uninitialized rather than opened via a creating call.
pub async fn check_dataset(dataset_path: &Path) -> anyhow::Result<Vec<Diagnostic>> {
    if !dataset_path.exists() {
        return Ok(vec![Diagnostic::new(
            "dataset.present",
            Severity::Ok,
            Some(format!("no dataset at {}", dataset_path.display())),
            None,
            format!("run `markq index {}` to create it", dataset_path.display()),
        )]);
    }

    let idx = match LanceIndex::open(dataset_path).await {
        Ok(idx) => idx,
        Err(err) => {
            return Ok(vec![Diagnostic::error(
                "dataset.present",
                err.to_string(),
                "a dataset markq can open",
                "inspect the dataset directory; if it is corrupt, remove it and run `markq index` to rebuild it",
            )]);
        }
    };

    let mut findings = vec![Diagnostic::ok("dataset.present")];
    let metadata = idx.metadata().await.context("read dataset metadata")?;
    findings.push(check_schema_version(metadata.schema_version));
    findings.push(check_embedder(&metadata));
    let health = idx.health().await.context("read index health")?;
    findings.extend(check_indexes(&health));
    Ok(findings)
}

/// Compare the dataset's recorded schema version against what this binary
/// supports. Backward drift (dataset older than the binary) is cleared by
/// rebuilding the dataset; forward drift (dataset newer than the binary
/// knows) is cleared by upgrading markq, never by rebuilding data the
/// current binary can't fully interpret.
fn check_schema_version(dataset_version: u32) -> Diagnostic {
    match dataset_version.cmp(&SCHEMA_VERSION) {
        Ordering::Equal => Diagnostic::ok("dataset.schema_version"),
        Ordering::Less => Diagnostic::error(
            "dataset.schema_version",
            dataset_version.to_string(),
            SCHEMA_VERSION.to_string(),
            format!("run `markq index` to rebuild the dataset at schema_version {SCHEMA_VERSION}"),
        ),
        Ordering::Greater => Diagnostic::error(
            "dataset.schema_version",
            dataset_version.to_string(),
            SCHEMA_VERSION.to_string(),
            "upgrade markq to a version that supports this dataset's schema_version".to_string(),
        ),
    }
}

/// Compare the dataset's recorded embedder identity and dimension against
/// the configured embedder. A dataset that has never been embedded records
/// neither and has nothing to compare, which is not a problem on its own.
fn check_embedder(metadata: &DatasetMetadata) -> Diagnostic {
    let configured_model = ModelRole::Embedder.known_model().id();
    let configured_dim = EMBEDDING_DIM_DEFAULT as u32;

    let (Some(recorded_model), Some(recorded_dim)) =
        (&metadata.embedder_model, metadata.embedder_dim)
    else {
        return Diagnostic::ok("dataset.embedder");
    };

    if recorded_dim != configured_dim {
        return Diagnostic::error(
            "dataset.embedder",
            format!("dim={recorded_dim}"),
            format!("dim={configured_dim}"),
            "re-embed the dataset with `markq embed` using the configured embedder, or reconfigure the embedder to match".to_string(),
        );
    }
    if recorded_model != configured_model {
        return Diagnostic::error(
            "dataset.embedder",
            format!("model={recorded_model}"),
            format!("model={configured_model}"),
            "re-embed the dataset with `markq embed` using the configured embedder, or reconfigure the embedder to match".to_string(),
        );
    }
    Diagnostic::ok("dataset.embedder")
}

/// The lexical and vector index findings (FR-018a): coverage is reported as
/// indexed/unindexed row counts against the current row count, never a
/// present/absent boolean, so a partially rebuilt index is distinguishable
/// from a fully missing one.
fn check_indexes(health: &IndexHealth) -> Vec<Diagnostic> {
    vec![
        check_one_index(health, ChunkColumn::TEXT, "index.lexical", "markq index"),
        check_one_index(
            health,
            ChunkColumn::EMBEDDING,
            "index.vector",
            "markq embed",
        ),
    ]
}

fn check_one_index(
    health: &IndexHealth,
    column: &str,
    check: &str,
    rebuild_command: &str,
) -> Diagnostic {
    if health.row_count == 0 {
        return Diagnostic::ok(check);
    }
    match health.indexes.iter().find(|c| c.column == column) {
        None => Diagnostic::error(
            check,
            "no index built",
            format!("{} row(s) indexed", health.row_count),
            format!("run `{rebuild_command}` to build the index"),
        ),
        Some(coverage) if coverage.unindexed_rows > 0 => Diagnostic::warning(
            check,
            format!(
                "{} of {} row(s) indexed",
                coverage.indexed_rows, health.row_count
            ),
            format!("{} row(s) indexed", health.row_count),
            format!("run `{rebuild_command}` to cover the remaining rows"),
        ),
        Some(_) => Diagnostic::ok(check),
    }
}

/// Every known role's model-cache finding (FR-018).
pub fn check_models(models_dir: &Path) -> anyhow::Result<Vec<Diagnostic>> {
    let mut findings = Vec::new();
    for role in ModelRole::ALL {
        findings.extend(check_model(role, models_dir)?);
    }
    Ok(findings)
}

/// One role's model-cache finding: presence/completeness/verification, plus
/// a separate finding when its download lock is currently held (FR-020: this
/// never loads the model; `CacheEntry::inspect` only stats files).
fn check_model(role: ModelRole, models_dir: &Path) -> anyhow::Result<Vec<Diagnostic>> {
    let check = format!("models.{}", role.as_str());
    let model_path = models_dir.join(role.known_model().filename());
    let mut findings = Vec::new();

    if DigestSidecar::is_stale_for(&model_path) {
        findings.push(Diagnostic::warning(
            &check,
            "a digest record with no matching file (stale sidecar)",
            "a verified file matching its digest record",
            format!(
                "run `markq models pull {}` to fetch it again",
                role.as_str()
            ),
        ));
    } else {
        let entry = CacheEntry::inspect(role, models_dir)
            .with_context(|| format!("inspect {} cache entry", role.as_str()))?;
        findings.push(match entry.state {
            CacheState::Absent => Diagnostic::error(
                &check,
                "not cached",
                "cached and verified",
                format!("run `markq models pull {}` to fetch it", role.as_str()),
            ),
            CacheState::Partial { bytes } => Diagnostic::warning(
                &check,
                format!("incomplete download ({bytes} byte(s) so far)"),
                "a complete, verified file",
                format!(
                    "run `markq models pull {}` to resume the download",
                    role.as_str()
                ),
            ),
            CacheState::Unverified => Diagnostic::warning(
                &check,
                "present but not verified",
                "a verified file",
                format!("run `markq models pull {}` to verify it", role.as_str()),
            ),
            CacheState::Verified => Diagnostic::ok(&check),
        });
    }

    if model_cache::download_in_progress(&model_path)? {
        findings.push(Diagnostic::warning(
            format!("{check}.download"),
            "a download is currently in progress",
            "no in-progress download",
            format!(
                "wait for the `markq models pull {}` in progress to finish, then re-run `markq doctor`",
                role.as_str()
            ),
        ));
    }

    Ok(findings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use markq_core::{DatasetMetadata, IndexCoverage, SCHEMA_VERSION};

    #[tokio::test]
    async fn uninitialized_dataset_is_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let dataset_path = dir.path().join("chunks.lance");

        let findings = check_dataset(&dataset_path).await.unwrap();

        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].check, "dataset.present");
        assert_eq!(findings[0].severity, Severity::Ok);
        assert!(
            !findings[0].next_step.is_empty(),
            "an uninitialized dataset must still name a next step"
        );
    }

    #[test]
    fn schema_drift_names_both_versions() {
        let dataset_version = 0;
        let d = check_schema_version(dataset_version);
        assert_eq!(d.severity, Severity::Error);
        assert_eq!(d.observed.as_deref(), Some("0"));
        assert_eq!(
            d.expected.as_deref(),
            Some(SCHEMA_VERSION.to_string().as_str())
        );
    }

    #[test]
    fn forward_drift_suggests_upgrade() {
        let d = check_schema_version(SCHEMA_VERSION + 1);
        assert_eq!(d.severity, Severity::Error);
        assert!(
            d.next_step.to_ascii_lowercase().contains("upgrade"),
            "forward drift should point at upgrading markq, got: {}",
            d.next_step
        );
    }

    #[test]
    fn embedder_dim_mismatch_names_both_values() {
        let metadata = DatasetMetadata {
            schema_version: SCHEMA_VERSION,
            lance_manifest_version: 1,
            lance_file_format_version: "2.1".to_string(),
            lancedb_crate_version: "0.27.2".to_string(),
            embedder_model: Some(
                markq_inference::model_cache::ModelRole::Embedder
                    .known_model()
                    .id()
                    .to_string(),
            ),
            embedder_dim: Some(4),
        };

        let d = check_embedder(&metadata);

        assert_eq!(d.severity, Severity::Error);
        assert!(d.observed.as_deref().unwrap().contains('4'));
        assert!(d.expected.as_deref().unwrap().contains("1024"));
    }

    #[test]
    fn missing_index_names_the_rebuild_command() {
        let health = IndexHealth {
            row_count: 4,
            indexes: vec![],
        };
        let findings = check_indexes(&health);
        let lexical = findings
            .iter()
            .find(|f| f.check == "index.lexical")
            .unwrap();
        assert_eq!(lexical.severity, Severity::Error);
        assert!(
            lexical.next_step.contains("markq index"),
            "got: {}",
            lexical.next_step
        );
        let vector = findings.iter().find(|f| f.check == "index.vector").unwrap();
        assert_eq!(vector.severity, Severity::Error);
        assert!(
            vector.next_step.contains("markq embed"),
            "got: {}",
            vector.next_step
        );
    }

    #[test]
    fn partial_coverage_reports_counts_not_a_boolean() {
        let health = IndexHealth {
            row_count: 10,
            indexes: vec![IndexCoverage {
                column: ChunkColumn::EMBEDDING.to_string(),
                indexed_rows: 6,
                unindexed_rows: 4,
            }],
        };
        let findings = check_indexes(&health);
        let vector = findings.iter().find(|f| f.check == "index.vector").unwrap();
        assert_eq!(vector.severity, Severity::Warning);
        let observed = vector.observed.as_deref().unwrap();
        assert!(observed.contains('6'), "got: {observed}");
        assert!(observed.contains("10"), "got: {observed}");
    }

    #[test]
    fn empty_dataset_reports_zero_rows_cleanly() {
        let health = IndexHealth {
            row_count: 0,
            indexes: vec![],
        };
        let findings = check_indexes(&health);
        for finding in &findings {
            assert_eq!(
                finding.severity,
                Severity::Ok,
                "check {} should be ok on an empty dataset",
                finding.check
            );
        }
    }

    #[test]
    fn missing_model_names_the_pull_command() {
        let dir = tempfile::tempdir().unwrap();

        let findings = check_model(ModelRole::Embedder, dir.path()).unwrap();

        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].check, "models.embedder");
        assert_eq!(findings[0].severity, Severity::Error);
        assert!(
            findings[0].next_step.contains("markq models pull embedder"),
            "got: {}",
            findings[0].next_step
        );
    }

    #[test]
    fn partial_model_is_reported_resumable() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir
            .path()
            .join(ModelRole::Embedder.known_model().filename());
        std::fs::write(
            markq_inference::model_cache::partial_path_for(&model_path),
            b"0123456789",
        )
        .unwrap();

        let findings = check_model(ModelRole::Embedder, dir.path()).unwrap();

        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, Severity::Warning);
        assert!(
            findings[0].next_step.contains("resume"),
            "got: {}",
            findings[0].next_step
        );
    }

    #[test]
    fn stale_sidecar_is_not_verified() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir
            .path()
            .join(ModelRole::Embedder.known_model().filename());
        markq_inference::model_cache::DigestSidecar {
            digest: "abc".into(),
            size_bytes: 5,
            source: "hub lfs metadata".into(),
        }
        .write_for(&model_path)
        .unwrap();

        let findings = check_model(ModelRole::Embedder, dir.path()).unwrap();

        assert_eq!(findings.len(), 1);
        assert_ne!(
            findings[0].severity,
            Severity::Ok,
            "a stale sidecar with no matching file must not read as verified"
        );
    }

    #[test]
    fn held_lock_is_reported_as_in_progress() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir
            .path()
            .join(ModelRole::Embedder.known_model().filename());
        let content = b"hello";
        std::fs::write(&model_path, content).unwrap();
        markq_inference::model_cache::DigestSidecar {
            digest: markq_inference::sha256_hex(&model_path).unwrap(),
            size_bytes: content.len() as u64,
            source: "hub lfs metadata".into(),
        }
        .write_for(&model_path)
        .unwrap();

        let lock_path = markq_inference::model_cache::lock_path_for(&model_path);
        let lock_file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
            .unwrap();
        fs4::FileExt::try_lock_exclusive(&lock_file).unwrap();

        let findings = check_model(ModelRole::Embedder, dir.path()).unwrap();

        fs4::FileExt::unlock(&lock_file).unwrap();

        assert_eq!(findings.len(), 2);
        assert_eq!(
            findings[0].severity,
            Severity::Ok,
            "the file itself is fine"
        );
        assert_eq!(findings[1].severity, Severity::Warning);
        assert!(
            findings[1]
                .observed
                .as_deref()
                .unwrap()
                .contains("progress"),
            "got: {:?}",
            findings[1].observed
        );
    }

    #[test]
    fn unheld_lock_file_is_not_reported_as_a_problem() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir
            .path()
            .join(ModelRole::Embedder.known_model().filename());
        let content = b"hello";
        std::fs::write(&model_path, content).unwrap();
        markq_inference::model_cache::DigestSidecar {
            digest: markq_inference::sha256_hex(&model_path).unwrap(),
            size_bytes: content.len() as u64,
            source: "hub lfs metadata".into(),
        }
        .write_for(&model_path)
        .unwrap();
        // A lock file left on disk (e.g. by a process that exited normally)
        // but not currently held by anyone.
        std::fs::write(
            markq_inference::model_cache::lock_path_for(&model_path),
            b"",
        )
        .unwrap();

        let findings = check_model(ModelRole::Embedder, dir.path()).unwrap();

        assert_eq!(
            findings.len(),
            1,
            "an unheld lock file must not add a finding"
        );
        assert_eq!(findings[0].severity, Severity::Ok);
    }

    #[test]
    fn any_error_finding_yields_failure_status() {
        let all_ok = vec![
            Diagnostic::ok("dataset.present"),
            Diagnostic::warning("models.reranker", "unverified", "verified", "run pull"),
        ];
        assert!(
            !any_error(&all_ok),
            "warnings alone must not fail the exit status"
        );

        let with_error = vec![
            Diagnostic::ok("dataset.present"),
            Diagnostic::error("models.embedder", "absent", "verified", "run pull"),
        ];
        assert!(any_error(&with_error));
    }

    #[test]
    fn ok_findings_carry_no_next_step() {
        let d = Diagnostic::ok("dataset.present");
        assert_eq!(d.severity, Severity::Ok);
        assert_eq!(d.next_step, "");
        assert_eq!(d.observed, None);
        assert_eq!(d.expected, None);
    }
}
