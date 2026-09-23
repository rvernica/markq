//! GGUF model resolution and cached download. Hugging Face is the source of
//! record, and files land flat in `~/.cache/markq/models/<filename>` (or
//! `$MARKQ_MODELS_DIR` when set).

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use hf_hub::HFClient;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::info;

/// Curated set of GGUFs markq knows how to fetch. Each new model that's
/// needed adds an entry. v1 ships the embedder and a cross-encoder reranker;
/// a HyDE generator will join this list later.
#[derive(Debug, Clone, Copy)]
pub enum KnownModel {
    /// Qwen3-Embedding-0.6B, Q8_0 quantization (the smallest GGUF the upstream
    /// repo publishes — no Q4_K_M variant exists at the time of writing).
    Qwen3Embedding06B,
    /// Qwen3-Reranker-0.6B, Q8_0 quantization. Used for cross-encoder reranking
    /// in hybrid search.
    Qwen3Reranker06B,
}

impl KnownModel {
    /// HF repo `<owner>/<name>` containing the GGUF asset.
    pub fn repo(self) -> (&'static str, &'static str) {
        match self {
            KnownModel::Qwen3Embedding06B => ("Qwen", "Qwen3-Embedding-0.6B-GGUF"),
            KnownModel::Qwen3Reranker06B => ("ggml-org", "Qwen3-Reranker-0.6B-Q8_0-GGUF"),
        }
    }

    /// Filename within the repo.
    pub fn filename(self) -> &'static str {
        match self {
            KnownModel::Qwen3Embedding06B => "Qwen3-Embedding-0.6B-Q8_0.gguf",
            KnownModel::Qwen3Reranker06B => "qwen3-reranker-0.6b-q8_0.gguf",
        }
    }

    /// Stable string identifier for this model variant. Must not change across
    /// markq versions. For embedders, the identifier is recorded in the dataset's
    /// user metadata under `markq.embedder_model` and validated on dataset reopen.
    pub fn id(self) -> &'static str {
        match self {
            KnownModel::Qwen3Embedding06B => "Qwen/Qwen3-Embedding-0.6B-GGUF/Q8_0",
            KnownModel::Qwen3Reranker06B => "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/Q8_0",
        }
    }
}

/// What a verified download records next to its model file, so a later run can
/// re-check the file without asking the hub what "correct" means. It travels
/// with the directory when an operator copies a staged cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DigestSidecar {
    /// Lowercase hex SHA-256 of the verified file.
    pub digest: String,
    /// Size of the verified file, which is what the per-load check compares.
    pub size_bytes: u64,
    /// Where the digest came from, so a later disagreement is attributable.
    pub source: String,
}

impl DigestSidecar {
    /// The sidecar path for a model file.
    pub fn path_for(model_path: &Path) -> PathBuf {
        let mut name = model_path.as_os_str().to_os_string();
        name.push(".digest.json");
        PathBuf::from(name)
    }

    /// Read the sidecar beside `model_path`, if one is there. A sidecar that
    /// cannot be parsed is an error rather than a silent absence: it means
    /// something wrote the file, and treating that as "no record" would hide
    /// the problem behind a re-download.
    pub fn read_for(model_path: &Path) -> Result<Option<Self>> {
        let path = Self::path_for(model_path);
        match std::fs::read_to_string(&path) {
            Ok(raw) => {
                let sidecar = serde_json::from_str(&raw)
                    .with_context(|| format!("parse digest sidecar {}", path.display()))?;
                Ok(Some(sidecar))
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err).with_context(|| format!("read {}", path.display())),
        }
    }

    /// Write the sidecar beside `model_path` by rename, so a reader never sees
    /// a half-written record.
    pub fn write_for(&self, model_path: &Path) -> Result<()> {
        let path = Self::path_for(model_path);
        let mut temporary = path.clone();
        temporary.set_extension("tmp");
        let encoded = serde_json::to_vec_pretty(self).context("encode digest sidecar")?;
        std::fs::write(&temporary, encoded)
            .with_context(|| format!("write {}", temporary.display()))?;
        std::fs::rename(&temporary, &path)
            .with_context(|| format!("rename into {}", path.display()))?;
        Ok(())
    }

    /// A sidecar whose model file is gone describes nothing, and must not let a
    /// later download skip its own verification.
    pub fn is_stale_for(model_path: &Path) -> bool {
        Self::path_for(model_path).exists() && !model_path.exists()
    }
}

/// The name a user types to refer to one of the models markq can fetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelRole {
    Embedder,
    Reranker,
}

impl ModelRole {
    /// Every role, in the order `markq models ls` presents them.
    pub const ALL: [ModelRole; 2] = [ModelRole::Embedder, ModelRole::Reranker];

    pub fn as_str(self) -> &'static str {
        match self {
            ModelRole::Embedder => "embedder",
            ModelRole::Reranker => "reranker",
        }
    }

    pub fn known_model(self) -> KnownModel {
        match self {
            ModelRole::Embedder => KnownModel::Qwen3Embedding06B,
            ModelRole::Reranker => KnownModel::Qwen3Reranker06B,
        }
    }

    /// The role list an unknown-role error offers the user.
    fn valid_list() -> String {
        ModelRole::ALL
            .iter()
            .map(|role| role.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl std::str::FromStr for ModelRole {
    type Err = markq_core::Error;

    fn from_str(name: &str) -> std::result::Result<Self, Self::Err> {
        match name {
            "embedder" => Ok(ModelRole::Embedder),
            "reranker" => Ok(ModelRole::Reranker),
            other => Err(markq_core::Error::UnknownModelRole {
                given: other.to_string(),
                valid: ModelRole::valid_list(),
            }),
        }
    }
}

/// The directory markq caches model files in. Honors `$MARKQ_MODELS_DIR`,
/// then falls back to `$XDG_CACHE_HOME/markq/models`, then `~/.cache/markq/
/// models`, then `./markq/models` if no cache dir can be resolved (containers,
/// CI without `$HOME`).
pub fn models_dir() -> PathBuf {
    if let Ok(p) = std::env::var("MARKQ_MODELS_DIR") {
        return PathBuf::from(p);
    }
    let mut p = dirs::cache_dir().unwrap_or_else(|| PathBuf::from("."));
    p.push("markq");
    p.push("models");
    p
}

/// Resolve a `KnownModel` to a local path, downloading if absent.
///
/// Downloads run on the current tokio runtime (the CLI's `#[tokio::main]`).
pub async fn ensure_model(model: KnownModel) -> Result<PathBuf> {
    let (owner, name) = model.repo();
    let filename = model.filename();
    let local_dir = models_dir();
    std::fs::create_dir_all(&local_dir)
        .with_context(|| format!("create models dir {}", local_dir.display()))?;

    let expected_path = local_dir.join(filename);
    if expected_path.exists() {
        info!(path = %expected_path.display(), "model already cached");
        return Ok(expected_path);
    }

    info!(
        repo = format!("{owner}/{name}"),
        file = filename,
        target = %expected_path.display(),
        "downloading model"
    );

    let client = HFClient::new().context("HFClient::new")?;
    let repo = client.model(owner, name);
    let downloaded = repo
        .download_file()
        .filename(filename)
        .local_dir(local_dir.clone())
        .send()
        .await
        .with_context(|| format!("hf-hub download_file {owner}/{name}/{filename}"))?;

    Ok(downloaded)
}

/// Verify `path`'s SHA-256 matches `expected` (hex, case-insensitive). Public
/// alongside `sha256_hex` so callers outside this module can check a file they
/// already have against a digest they obtained elsewhere.
pub fn verify_sha256_against(path: &Path, expected: &str) -> Result<()> {
    let actual = sha256_hex(path)?;
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(anyhow!(
            "SHA-256 mismatch for {}: expected {expected}, got {actual}",
            path.display()
        ));
    }
    Ok(())
}

/// Hex-encoded SHA-256 of a file. Exposed so tests can synthesize tampered
/// fixtures without rebuilding the verification logic.
pub fn sha256_hex(path: &Path) -> Result<String> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut r = BufReader::new(f);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = r
            .read(&mut buf)
            .with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn sha256_hex_known_value() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a");
        let mut f = File::create(&p).unwrap();
        f.write_all(b"hello").unwrap();
        drop(f);
        // sha256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        assert_eq!(
            sha256_hex(&p).unwrap(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn verify_sha256_against_accepts_match_rejects_tamper() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("blob");
        std::fs::write(&p, b"hello").unwrap();
        let good = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

        // Matching digest passes (case-insensitive).
        verify_sha256_against(&p, good).expect("matching sha should pass");
        verify_sha256_against(&p, &good.to_uppercase())
            .expect("case-insensitive compare should pass");

        // Any single-bit flip fails with a clear message.
        let bad = "0000000000000000000000000000000000000000000000000000000000000000";
        let err = verify_sha256_against(&p, bad).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("SHA-256 mismatch"), "got: {msg}");
        assert!(
            msg.contains(good),
            "error should report actual digest: {msg}"
        );
    }

    #[test]
    fn sidecar_round_trips_digest_size_and_source() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.gguf");
        std::fs::write(&model, b"hello").unwrap();

        let written = DigestSidecar {
            digest: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824".into(),
            size_bytes: 5,
            source: "hub lfs metadata".into(),
        };
        written.write_for(&model).unwrap();

        let read = DigestSidecar::read_for(&model)
            .unwrap()
            .expect("sidecar present");
        assert_eq!(read, written);
    }

    #[test]
    fn sidecar_without_its_model_file_is_stale() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.gguf");
        std::fs::write(&model, b"hello").unwrap();
        DigestSidecar {
            digest: "abc".into(),
            size_bytes: 5,
            source: "hub lfs metadata".into(),
        }
        .write_for(&model)
        .unwrap();

        assert!(!DigestSidecar::is_stale_for(&model));

        std::fs::remove_file(&model).unwrap();
        assert!(DigestSidecar::is_stale_for(&model));
    }

    #[test]
    fn unknown_role_error_lists_valid_roles() {
        let err = "embeder".parse::<ModelRole>().unwrap_err();
        let message = err.to_string();
        assert!(message.contains("embeder"), "got: {message}");
        assert!(message.contains("embedder"), "got: {message}");
        assert!(message.contains("reranker"), "got: {message}");
    }

    #[test]
    fn roles_round_trip_to_known_models() {
        for role in ModelRole::ALL {
            assert_eq!(role.as_str().parse::<ModelRole>().unwrap(), role);
        }
        assert_eq!(
            ModelRole::Embedder.known_model().id(),
            KnownModel::Qwen3Embedding06B.id()
        );
        assert_eq!(
            ModelRole::Reranker.known_model().id(),
            KnownModel::Qwen3Reranker06B.id()
        );
        assert_eq!(ModelRole::Embedder.as_str(), "embedder");
        assert_eq!(ModelRole::Reranker.as_str(), "reranker");
    }

    #[test]
    fn qwen3_reranker_06b_resolves_to_expected_cache_filename() {
        let filename = KnownModel::Qwen3Reranker06B.filename();
        assert_eq!(filename, "qwen3-reranker-0.6b-q8_0.gguf");

        let cache_path = models_dir().join(filename);
        let cache_path_str = cache_path.to_string_lossy();
        assert!(
            cache_path_str.ends_with("markq/models/qwen3-reranker-0.6b-q8_0.gguf"),
            "expected cache path to end with markq/models/qwen3-reranker-0.6b-q8_0.gguf, got: {}",
            cache_path_str
        );
    }
}
