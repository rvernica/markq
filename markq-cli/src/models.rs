//! `markq models pull` / `markq models ls`: fetch and inspect the GGUF model
//! cache without touching the retrieval dataset.

use std::path::Path;

use anyhow::{Context, Result};
use markq_inference::model_cache::{download_model, CacheEntry, CacheState, ModelRole};
use markq_inference::HFClient;
use serde::Serialize;

/// One requested role's outcome from `markq models pull`.
#[derive(Debug)]
pub struct PullOutcome {
    pub role: ModelRole,
    /// `false` when the role was already verified and nothing was fetched.
    pub transferred: bool,
    pub result: std::result::Result<CacheEntry, String>,
}

impl PullOutcome {
    pub fn is_ready(&self) -> bool {
        self.result.is_ok()
    }
}

/// Make every requested role ready: an already-verified role is a no-op
/// (FR-010); everything else is fetched and verified via `download_model`.
/// `role_name` selects one role; `None` processes every known role.
pub async fn run_pull(
    role_name: Option<&str>,
    models_dir: &Path,
    client: &HFClient,
) -> Result<Vec<PullOutcome>> {
    let roles = resolve_roles(role_name)?;
    let mut outcomes = Vec::with_capacity(roles.len());
    for role in roles {
        let before = CacheEntry::inspect(role, models_dir)
            .with_context(|| format!("inspect {} cache entry", role.as_str()))?;
        if before.state == CacheState::Verified {
            outcomes.push(PullOutcome {
                role,
                transferred: false,
                result: Ok(before),
            });
            continue;
        }

        let result = match download_model(client, role.known_model(), models_dir).await {
            Ok(_) => CacheEntry::inspect(role, models_dir).map_err(|err| err.to_string()),
            Err(err) => Err(err.to_string()),
        };
        outcomes.push(PullOutcome {
            role,
            transferred: true,
            result,
        });
    }
    Ok(outcomes)
}

/// The roles `pull`/`ls` should process: the one the user named, or every
/// known role when none was given. An unknown name surfaces the same
/// `UnknownModelRole` error `ModelRole::from_str` produces (FR-003).
fn resolve_roles(role_name: Option<&str>) -> Result<Vec<ModelRole>> {
    match role_name {
        Some(name) => Ok(vec![name.parse::<ModelRole>()?]),
        None => Ok(ModelRole::ALL.to_vec()),
    }
}

/// One role's row in `markq models ls`: exactly the fields FR-004 requires,
/// present or not.
#[derive(Debug, Serialize)]
pub struct ModelRow {
    pub role: String,
    pub model_id: String,
    pub filename: String,
    pub state: String,
    pub size_bytes: Option<u64>,
    pub digest: Option<String>,
    pub digest_source: Option<String>,
}

impl ModelRow {
    fn from_entry(entry: CacheEntry) -> Self {
        let state = match entry.state {
            CacheState::Absent => "absent",
            CacheState::Partial { .. } => "partial",
            CacheState::Unverified => "unverified",
            CacheState::Verified => "verified",
        }
        .to_string();
        Self {
            role: entry.role.as_str().to_string(),
            model_id: entry.role.known_model().id().to_string(),
            filename: entry.role.known_model().filename().to_string(),
            state,
            size_bytes: entry.size_bytes,
            digest: entry.digest,
            digest_source: entry.digest_source,
        }
    }
}

/// One row per known role, whether or not it is cached (FR-004).
pub fn ls_rows(models_dir: &Path) -> Result<Vec<ModelRow>> {
    ModelRole::ALL
        .into_iter()
        .map(|role| {
            CacheEntry::inspect(role, models_dir)
                .with_context(|| format!("inspect {} cache entry", role.as_str()))
                .map(ModelRow::from_entry)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use markq_inference::model_cache::{DigestSidecar, KnownModel};

    #[test]
    fn ls_lists_every_role_including_absent_ones() {
        let dir = tempfile::tempdir().unwrap();

        let rows = ls_rows(dir.path()).unwrap();

        assert_eq!(rows.len(), ModelRole::ALL.len());
        for row in &rows {
            assert_eq!(row.state, "absent");
            assert_eq!(row.size_bytes, None);
            assert_eq!(row.digest, None);
        }
        let names: Vec<&str> = rows.iter().map(|row| row.role.as_str()).collect();
        assert!(names.contains(&"embedder"), "got: {names:?}");
        assert!(names.contains(&"reranker"), "got: {names:?}");
    }

    #[test]
    fn ls_json_carries_state_size_digest_and_source() {
        let dir = tempfile::tempdir().unwrap();
        let model = KnownModel::Qwen3Embedding06B;
        let target = dir.path().join(model.filename());
        std::fs::write(&target, b"hello").unwrap();
        DigestSidecar {
            digest: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824".into(),
            size_bytes: 5,
            source: "hub lfs metadata".into(),
        }
        .write_for(&target)
        .unwrap();

        let rows = ls_rows(dir.path()).unwrap();
        let value = serde_json::to_value(&rows).unwrap();
        let embedder = value
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["role"] == "embedder")
            .expect("embedder row present");

        assert_eq!(embedder["state"], "verified");
        assert_eq!(embedder["size_bytes"], 5);
        assert_eq!(
            embedder["digest"],
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        assert_eq!(embedder["digest_source"], "hub lfs metadata");
    }
}
