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
/// whenever `severity` is not `Ok`; an `Ok` finding may still carry
/// one: "not yet initialized, run `markq index`" is `Ok` rather than an
/// error, and still says what to do.
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
/// finding is `Severity::Error`. Warnings alone do not fail.
pub fn any_error(findings: &[Diagnostic]) -> bool {
    findings.iter().any(|f| f.severity == Severity::Error)
}

/// The complete `markq doctor` pass: dataset checks (presence, schema
/// version, embedder identity, index coverage) followed by every known
/// role's model-cache check. Read-only throughout and never loads a
/// model; works whether or not the dataset has ever been created
/// and requires no network access, since every check reads
/// only local state.
pub async fn run_doctor(
    dataset_path: &Path,
    models_dir: &Path,
    verify: bool,
) -> anyhow::Result<Vec<Diagnostic>> {
    let mut findings = check_dataset(dataset_path).await?;
    findings.extend(check_models(models_dir, verify)?);
    Ok(findings)
}

/// Dataset presence, schema version, and embedder identity.
/// Never creates a dataset: a path that does not exist is reported
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

/// The lexical and vector index findings: coverage is reported as
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
        None => Diagnostic::warning(
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

/// Every known role's model-cache finding. `verify` additionally
/// hashes each cached file's full content; without it, an already
/// cached role is checked by size only.
pub fn check_models(models_dir: &Path, verify: bool) -> anyhow::Result<Vec<Diagnostic>> {
    let mut findings = Vec::new();
    for role in ModelRole::ALL {
        findings.extend(check_model(role, models_dir, verify)?);
    }
    Ok(findings)
}

/// One role's model-cache finding: presence/completeness/verification, plus
/// a separate finding when its download lock is currently held (this
/// never loads the model; `CacheEntry::inspect` only stats files, and hashing
/// under `verify` reads file bytes, never loads them as a model).
fn check_model(
    role: ModelRole,
    models_dir: &Path,
    verify: bool,
) -> anyhow::Result<Vec<Diagnostic>> {
    let check = format!("models.{}", role.as_str());
    let model_path = models_dir.join(role.known_model().filename());
    let mut findings = Vec::new();

    let entry = CacheEntry::inspect(role, models_dir)
        .with_context(|| format!("inspect {} cache entry", role.as_str()))?;
    let sidecar = DigestSidecar::read_for(&model_path)?;

    {
        let mut finding = match entry.state {
            CacheState::Absent => Diagnostic::error(
                &check,
                "not cached",
                "cached and verified",
                format!("run `markq models pull {}` to fetch it", role.as_str()),
            ),
            // An unfinished transfer looks the same here whether it was
            // interrupted or discarded for failing its digest check, so this
            // says what is on disk rather than asserting a cause. Either way
            // `models pull` is the command that finishes it.
            CacheState::Partial { bytes } => Diagnostic::warning(
                &check,
                format!("an unfinished transfer ({bytes} byte(s) on disk)"),
                "a complete, verified file",
                format!("run `markq models pull {}` to finish it", role.as_str()),
            ),
            CacheState::Unverified => match &sidecar {
                Some(record) => Diagnostic::error(
                    &check,
                    format!(
                        "{} byte(s) on disk, but its digest record says {}",
                        entry.size_bytes.unwrap_or_default(),
                        record.size_bytes
                    ),
                    "a file matching its digest record",
                    format!(
                        "delete {} and run `markq models pull {}` to fetch it again",
                        model_path.display(),
                        role.as_str()
                    ),
                ),
                None => Diagnostic::warning(
                    &check,
                    if verify {
                        // Saying nothing here would let a full-verification run
                        // look like it had examined a file it never hashed.
                        "present but not verified, and there is no recorded digest to check it against"
                    } else {
                        "present but not verified"
                    },
                    "a verified file",
                    format!("run `markq models pull {}` to verify it", role.as_str()),
                ),
            },
            CacheState::Verified => Diagnostic::ok(&check),
        };

        // The size-only check above can't see a same-length alteration; a
        // full hash catches it, but only when explicitly requested.
        if verify && entry.state == CacheState::Verified {
            if let Some(sidecar) = DigestSidecar::read_for(&model_path)? {
                let actual = markq_inference::sha256_hex(&model_path)?;
                if !actual.eq_ignore_ascii_case(&sidecar.digest) {
                    finding = Diagnostic::error(
                        &check,
                        format!("full digest {actual} does not match its recorded digest"),
                        format!("digest {}", sidecar.digest),
                        format!(
                            "delete {} and run `markq models pull {}` to fetch it again",
                            model_path.display(),
                            role.as_str()
                        ),
                    );
                }
            }
        }

        findings.push(finding);
    }

    // A record whose file is gone explains itself, but it must not soften the
    // finding above: a missing model is still missing.
    if DigestSidecar::is_stale_for(&model_path) {
        findings.push(Diagnostic::warning(
            format!("{check}.record"),
            "a digest record with no matching file",
            "a digest record describing a file that is there",
            format!(
                "run `markq models pull {}` to fetch the file its record describes",
                role.as_str()
            ),
        ));
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
    use markq_inference::model_cache::{lock_path_for, partial_path_for};
    use markq_inference::sha256_hex;

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
            embedder_model: Some(ModelRole::Embedder.known_model().id().to_string()),
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
        // Not an error: a corpus that has been indexed but never embedded is a
        // supported state, and grading it an error fails the exit status a
        // script gates on for an installation that works.
        assert_eq!(lexical.severity, Severity::Warning);
        assert!(
            lexical.next_step.contains("markq index"),
            "got: {}",
            lexical.next_step
        );
        let vector = findings.iter().find(|f| f.check == "index.vector").unwrap();
        assert_eq!(vector.severity, Severity::Warning);
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

        let findings = check_model(ModelRole::Embedder, dir.path(), false).unwrap();

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
        std::fs::write(partial_path_for(&model_path), b"0123456789").unwrap();

        let findings = check_model(ModelRole::Embedder, dir.path(), false).unwrap();

        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, Severity::Warning);
        assert!(
            findings[0].next_step.contains("markq models pull"),
            "got: {}",
            findings[0].next_step
        );
        // The byte count is the useful part, and it must not claim a cause the
        // local state cannot distinguish.
        assert!(
            findings[0]
                .observed
                .as_deref()
                .unwrap_or_default()
                .contains("10 byte(s)"),
            "got: {:?}",
            findings[0].observed
        );
    }

    #[test]
    fn stale_sidecar_is_not_verified() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir
            .path()
            .join(ModelRole::Embedder.known_model().filename());
        DigestSidecar {
            digest: "abc".into(),
            size_bytes: 5,
            source: "hub lfs metadata".into(),
        }
        .write_for(&model_path)
        .unwrap();

        let findings = check_model(ModelRole::Embedder, dir.path(), false).unwrap();

        // The leftover record explains itself, but the model is still missing,
        // and a stale record must not soften that into a warning a CI gate
        // would pass.
        let state = findings
            .iter()
            .find(|f| f.check == "models.embedder")
            .expect("a finding about the model itself");
        assert_eq!(state.severity, Severity::Error);
        let record = findings
            .iter()
            .find(|f| f.check == "models.embedder.record")
            .expect("a finding about the leftover record");
        assert_eq!(record.severity, Severity::Warning);
    }

    #[test]
    fn size_disagreeing_with_its_record_is_an_error_naming_the_repair() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir
            .path()
            .join(ModelRole::Embedder.known_model().filename());
        std::fs::write(&model_path, b"truncated").unwrap();
        DigestSidecar {
            digest: "abc".into(),
            size_bytes: 4096,
            source: "hub lfs metadata".into(),
        }
        .write_for(&model_path)
        .unwrap();

        let findings = check_model(ModelRole::Embedder, dir.path(), false).unwrap();
        let finding = findings
            .iter()
            .find(|f| f.check == "models.embedder")
            .unwrap();

        // The next load refuses this file outright, so predicting that is the
        // whole point of the diagnosis.
        assert_eq!(finding.severity, Severity::Error);
        // `models pull` skips a file it considers cached, so the advice has to
        // get the file out of the way first.
        assert!(
            finding.next_step.contains("delete"),
            "got: {}",
            finding.next_step
        );
    }

    #[test]
    fn verified_digest_mismatch_says_to_delete_before_pulling() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir
            .path()
            .join(ModelRole::Embedder.known_model().filename());
        std::fs::write(&model_path, b"hello").unwrap();
        DigestSidecar {
            digest: "0000000000000000000000000000000000000000000000000000000000000000".into(),
            size_bytes: 5,
            source: "hub lfs metadata".into(),
        }
        .write_for(&model_path)
        .unwrap();

        let findings = check_model(ModelRole::Embedder, dir.path(), true).unwrap();
        let finding = findings
            .iter()
            .find(|f| f.check == "models.embedder")
            .unwrap();

        assert_eq!(finding.severity, Severity::Error);
        // Pulling alone is a no-op here: the file still passes the size check
        // that `models pull` uses to decide a role is already done.
        assert!(
            finding.next_step.contains("delete"),
            "got: {}",
            finding.next_step
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
        DigestSidecar {
            digest: sha256_hex(&model_path).unwrap(),
            size_bytes: content.len() as u64,
            source: "hub lfs metadata".into(),
        }
        .write_for(&model_path)
        .unwrap();

        let lock_path = lock_path_for(&model_path);
        let lock_file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
            .unwrap();
        fs4::FileExt::try_lock_exclusive(&lock_file).unwrap();

        let findings = check_model(ModelRole::Embedder, dir.path(), false).unwrap();

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
        DigestSidecar {
            digest: sha256_hex(&model_path).unwrap(),
            size_bytes: content.len() as u64,
            source: "hub lfs metadata".into(),
        }
        .write_for(&model_path)
        .unwrap();
        // A lock file left on disk (e.g. by a process that exited normally)
        // but not currently held by anyone.
        std::fs::write(lock_path_for(&model_path), b"").unwrap();

        let findings = check_model(ModelRole::Embedder, dir.path(), false).unwrap();

        assert_eq!(
            findings.len(),
            1,
            "an unheld lock file must not add a finding"
        );
        assert_eq!(findings[0].severity, Severity::Ok);
    }

    #[test]
    fn verify_flag_detects_same_size_alteration() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir
            .path()
            .join(ModelRole::Embedder.known_model().filename());
        let original = b"hello";
        std::fs::write(&model_path, original).unwrap();
        DigestSidecar {
            digest: sha256_hex(&model_path).unwrap(),
            size_bytes: original.len() as u64,
            source: "hub lfs metadata".into(),
        }
        .write_for(&model_path)
        .unwrap();

        // Same length, different bytes: the size-only check can't see this.
        std::fs::write(&model_path, b"jello").unwrap();

        let default_findings = check_model(ModelRole::Embedder, dir.path(), false).unwrap();
        assert_eq!(
            default_findings[0].severity,
            Severity::Ok,
            "a same-size alteration is invisible to the size-only check"
        );

        let verified_findings = check_model(ModelRole::Embedder, dir.path(), true).unwrap();
        assert_eq!(
            verified_findings[0].severity,
            Severity::Error,
            "--verify must catch a same-size alteration the size check misses"
        );
    }

    #[test]
    fn default_run_does_not_hash() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir
            .path()
            .join(ModelRole::Embedder.known_model().filename());
        let original = b"hello";
        std::fs::write(&model_path, original).unwrap();
        DigestSidecar {
            digest: sha256_hex(&model_path).unwrap(),
            size_bytes: original.len() as u64,
            source: "hub lfs metadata".into(),
        }
        .write_for(&model_path)
        .unwrap();
        std::fs::write(&model_path, b"jello").unwrap();

        // A digest mismatch that only a hash would catch still reads as
        // verified without --verify, proving the default path checks size
        // only.
        let findings = check_model(ModelRole::Embedder, dir.path(), false).unwrap();
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
