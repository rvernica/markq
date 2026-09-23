//! `markq models pull` / `markq models ls`: fetch and inspect the GGUF model
//! cache without touching the retrieval dataset.

use std::path::Path;

use anyhow::{Context, Result};
use markq_inference::model_cache::{download_model, CacheEntry, CacheState, ModelRole};
use markq_inference::HFClient;

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
