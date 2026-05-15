//! GGUF model resolution + cached download. Hugging Face is the source of
//! record; `hf-hub` writes into `~/.cache/markq/models/<repo>/<file>` (or
//! `$MARKQ_MODELS_DIR` when set). After a download we optionally verify a
//! known SHA-256 against the file; this is best-effort in v1 — the full
//! resumable / `markq models pull` UX lands in .

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use hf_hub::HFClient;
use sha2::{Digest, Sha256};
use tracing::{info, warn};

/// Curated set of GGUFs markq knows how to fetch. Each milestone that needs a new
/// model adds an entry. v1 only ships the embedder; the reranker
/// and HyDE generator will join this list.
#[derive(Debug, Clone, Copy)]
pub enum KnownModel {
    /// Qwen3-Embedding-0.6B, Q8_0 quantization (the smallest GGUF the upstream
    /// repo publishes — no Q4_K_M variant exists at the time of writing).
    Qwen3Embedding06B,
}

impl KnownModel {
    /// HF repo `<owner>/<name>` containing the GGUF asset.
    pub fn repo(self) -> (&'static str, &'static str) {
        match self {
            KnownModel::Qwen3Embedding06B => ("Qwen", "Qwen3-Embedding-0.6B-GGUF"),
        }
    }

    /// Filename within the repo.
    pub fn filename(self) -> &'static str {
        match self {
            KnownModel::Qwen3Embedding06B => "Qwen3-Embedding-0.6B-Q8_0.gguf",
        }
    }

    /// Stable string identifier recorded in the dataset's user metadata under
    /// `markq.embedder_model`. The dimension validator on reopen compares
    /// against this exact value, so it must be stable across markq versions.
    pub fn id(self) -> &'static str {
        match self {
            KnownModel::Qwen3Embedding06B => "Qwen/Qwen3-Embedding-0.6B-GGUF/Q8_0",
        }
    }

    /// Known-good SHA-256 hex digest, when one has been recorded. `None` means
    /// "no pin yet"; the cache then logs a warning rather than failing the
    /// download. the `markq doctor` will surface unpinned models.
    pub fn sha256(self) -> Option<&'static str> {
        match self {
            // TODO: record the digest of the Qwen3-Embedding GGUF
            // after the first known-good download; until then the verifier
            // logs a warning and accepts whatever is on disk.
            KnownModel::Qwen3Embedding06B => None,
        }
    }
}

/// The directory markq caches model files in. Honors `$MARKQ_MODELS_DIR`,
/// then falls back to `$XDG_CACHE_HOME/markq/models`, then `~/.cache/markq/
/// models`, then `./.markq/models` if `$HOME` is unset.
pub fn models_dir() -> PathBuf {
    if let Ok(p) = std::env::var("MARKQ_MODELS_DIR") {
        return PathBuf::from(p);
    }
    let mut p = dirs::cache_dir().unwrap_or_else(|| PathBuf::from("./.markq"));
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
        verify_sha256_if_known(&expected_path, model)?;
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

    verify_sha256_if_known(&downloaded, model)?;
    Ok(downloaded)
}

/// Compute the SHA-256 of `path` and compare against `model.sha256()`. If the
/// model has no recorded digest, log a warning and return Ok.
fn verify_sha256_if_known(path: &Path, model: KnownModel) -> Result<()> {
    let Some(expected) = model.sha256() else {
        warn!(
            id = model.id(),
            path = %path.display(),
            "no SHA-256 pin recorded for this model; skipping verification"
        );
        return Ok(());
    };
    verify_sha256_against(path, expected)
}

/// Verify `path`'s SHA-256 matches `expected` (hex, case-insensitive).
/// Split out from `verify_sha256_if_known` so tests can drive the mismatch
/// branch without needing a `KnownModel` with a pinned digest.
fn verify_sha256_against(path: &Path, expected: &str) -> Result<()> {
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
}
