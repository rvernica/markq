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
use markq_inference::model_cache::ModelRole;
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
