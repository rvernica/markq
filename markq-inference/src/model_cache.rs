//! GGUF model resolution and cached download. Hugging Face is the source of
//! record, and files land flat in `~/.cache/markq/models/<filename>` (or
//! `$MARKQ_MODELS_DIR` when set).

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use fs4::FileExt;
use futures::StreamExt;
use hf_hub::repository::files::RepoTreeEntry;
use hf_hub::HFClient;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{info, warn};

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

/// Fetch `model` into `models_dir`, continuing an unfinished transfer rather
/// than starting over.
///
/// Bytes accumulate in a partial file beside the target and are renamed into
/// place only once the transfer is the size the repository declared, so a
/// reader never observes a short file under the model's own name.
pub async fn download_model(
    client: &HFClient,
    model: KnownModel,
    models_dir: &Path,
) -> Result<PathBuf> {
    std::fs::create_dir_all(models_dir)
        .with_context(|| format!("create models dir {}", models_dir.display()))?;
    let target = models_dir.join(model.filename());
    let partial = partial_path_for(&target);

    // Hold this model's lock for the rest of the function, so a second
    // caller fails immediately instead of racing this transfer (FR-010a,
    // FR-010b). The lock releases when `lock` drops at function return.
    let lock_path = lock_path_for(&target);
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("open lock {}", lock_path.display()))?;
    if let Err(err) = lock.try_lock_exclusive() {
        if err.kind() == std::io::ErrorKind::WouldBlock {
            return Err(markq_core::Error::ModelDownloadInProgress {
                role: role_for(model).as_str().to_string(),
                path: lock_path,
            }
            .into());
        }
        return Err(err).with_context(|| format!("lock {}", lock_path.display()));
    }

    let (owner, name) = model.repo();
    let repo = client.model(owner, name);
    let metadata = repo
        .get_file_metadata()
        .filepath(model.filename())
        .send()
        .await
        .with_context(|| format!("read metadata for {owner}/{name}/{}", model.filename()))?;
    let total = metadata.file_size;

    let mut have = match std::fs::metadata(&partial) {
        Ok(found) => found.len(),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => 0,
        Err(err) => return Err(err).with_context(|| format!("stat {}", partial.display())),
    };

    // A partial that already meets or exceeds the declared size cannot be a
    // prefix of the file being fetched, so it is discarded rather than resumed.
    if have >= total {
        if have > 0 {
            warn!(
                path = %partial.display(),
                have,
                total,
                "discarding an unfinished transfer that is not a prefix of the declared file"
            );
            std::fs::remove_file(&partial)
                .with_context(|| format!("remove {}", partial.display()))?;
        }
        have = 0;
    }

    let range = if have > 0 { Some(have..total) } else { None };
    let (_, mut stream) = repo
        .download_file_stream()
        .filename(model.filename())
        .maybe_range(range)
        .send()
        .await
        .with_context(|| format!("fetch {owner}/{name}/{}", model.filename()))?;

    let mut sink = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&partial)
        .with_context(|| format!("open {}", partial.display()))?;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("read body for {}", model.filename()))?;
        sink.write_all(&chunk)
            .with_context(|| format!("write {}", partial.display()))?;
    }
    sink.flush()
        .with_context(|| format!("flush {}", partial.display()))?;
    drop(sink);

    let written = std::fs::metadata(&partial)
        .with_context(|| format!("stat {}", partial.display()))?
        .len();
    if written != total {
        return Err(markq_core::Error::ModelFileIncomplete {
            path: partial,
            have: written,
            expected: total,
        }
        .into());
    }

    std::fs::rename(&partial, &target)
        .with_context(|| format!("rename into {}", target.display()))?;

    let declared = declared_digest(client, model)
        .await
        .with_context(|| format!("read declared digest for {}", model.filename()))?;
    verify_and_seal(&target, declared)?;

    Ok(target)
}

/// Hash a just-completed download and compare it against the digest the hub
/// declared for it, recording the result as a sidecar on a match.
///
/// A sidecar is written only once the hash matches (FR-008a): a mismatch
/// returns an error and leaves no sidecar, so the file stays on disk as
/// `Unverified` for inspection rather than being trusted. A repository that
/// declares no digest at all leaves the file unverified the same way, with
/// no error, since there is nothing to check it against (FR-011a).
fn verify_and_seal(target: &Path, declared_digest: Option<String>) -> Result<()> {
    let Some(expected) = declared_digest else {
        return Ok(());
    };

    let found = sha256_hex(target)?;
    if !found.eq_ignore_ascii_case(&expected) {
        return Err(markq_core::Error::ModelDigestMismatch {
            path: target.to_path_buf(),
            expected,
            found,
        }
        .into());
    }

    let size_bytes = std::fs::metadata(target)
        .with_context(|| format!("stat {}", target.display()))?
        .len();
    DigestSidecar {
        digest: found,
        size_bytes,
        source: "hub lfs metadata".to_string(),
    }
    .write_for(target)
}

/// The digest the repository declares for this model's file, when it declares
/// one. This is what a download is checked against; a repository that declares
/// nothing leaves the file unverifiable rather than untrusted.
pub async fn declared_digest(client: &HFClient, model: KnownModel) -> Result<Option<String>> {
    let (owner, name) = model.repo();
    let repo = client.model(owner, name);
    let wanted = model.filename();

    let entries = repo
        .list_tree()
        .expand(true)
        .send()
        .with_context(|| format!("list {owner}/{name}"))?;
    let mut entries = Box::pin(entries);

    while let Some(entry) = entries.next().await {
        let entry = entry.with_context(|| format!("read tree entry from {owner}/{name}"))?;
        if let RepoTreeEntry::File { path, lfs, .. } = entry {
            if path == wanted {
                return Ok(lfs.and_then(|info| info.sha256));
            }
        }
    }

    Ok(None)
}

/// Where an in-flight transfer accumulates bytes. Kept beside the model file so
/// the resume offset is simply this file's length.
pub fn partial_path_for(model_path: &Path) -> PathBuf {
    let mut name = model_path.as_os_str().to_os_string();
    name.push(".partial");
    PathBuf::from(name)
}

/// The advisory lock scoped to one model's download, so a download of one
/// model never blocks a download of another (FR-010a).
pub fn lock_path_for(model_path: &Path) -> PathBuf {
    let mut name = model_path.as_os_str().to_os_string();
    name.push(".lock");
    PathBuf::from(name)
}

/// The role that fetches `model`, for naming it in a contended-lock error.
/// Every `KnownModel` variant has exactly one corresponding role.
fn role_for(model: KnownModel) -> ModelRole {
    ModelRole::ALL
        .into_iter()
        .find(|role| role.known_model().filename() == model.filename())
        .expect("every KnownModel has a corresponding ModelRole")
}

/// What the cache knows about one role's file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheState {
    /// Neither the model file nor an unfinished transfer is present.
    Absent,
    /// An unfinished transfer is present; `bytes` is the resume offset.
    Partial { bytes: u64 },
    /// The file is here, but no sidecar vouches for it at its current size.
    Unverified,
    /// A sidecar records this file, and the file is still that size.
    Verified,
}

/// One role's state on disk, and the row `markq models ls` prints.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub role: ModelRole,
    pub path: PathBuf,
    pub state: CacheState,
    pub size_bytes: Option<u64>,
    pub digest: Option<String>,
    pub digest_source: Option<String>,
}

impl CacheEntry {
    /// Read one role's state from `models_dir` without touching the network.
    pub fn inspect(role: ModelRole, models_dir: &Path) -> Result<Self> {
        let path = models_dir.join(role.known_model().filename());
        let sidecar = DigestSidecar::read_for(&path)?;
        let size_bytes = match std::fs::metadata(&path) {
            Ok(metadata) => Some(metadata.len()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
            Err(err) => return Err(err).with_context(|| format!("stat {}", path.display())),
        };

        let state = match size_bytes {
            Some(size) => match &sidecar {
                Some(record) if record.size_bytes == size => CacheState::Verified,
                _ => CacheState::Unverified,
            },
            None => match std::fs::metadata(partial_path_for(&path)) {
                Ok(metadata) => CacheState::Partial {
                    bytes: metadata.len(),
                },
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => CacheState::Absent,
                Err(err) => {
                    return Err(err).with_context(|| format!("stat partial for {}", path.display()))
                }
            },
        };

        // A sidecar is only reported once it describes the file that is there.
        let (digest, digest_source) = match (&state, sidecar) {
            (CacheState::Verified, Some(record)) => (Some(record.digest), Some(record.source)),
            _ => (None, None),
        };

        Ok(Self {
            role,
            path,
            state,
            size_bytes,
            digest,
            digest_source,
        })
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
    fn cache_state_matches_files_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let role = ModelRole::Embedder;
        let model = dir.path().join(role.known_model().filename());

        let entry = CacheEntry::inspect(role, dir.path()).unwrap();
        assert_eq!(entry.state, CacheState::Absent);
        assert_eq!(entry.size_bytes, None);

        let partial = partial_path_for(&model);
        std::fs::write(&partial, b"0123456789").unwrap();
        let entry = CacheEntry::inspect(role, dir.path()).unwrap();
        assert_eq!(entry.state, CacheState::Partial { bytes: 10 });
        std::fs::remove_file(&partial).unwrap();

        std::fs::write(&model, b"hello").unwrap();
        let entry = CacheEntry::inspect(role, dir.path()).unwrap();
        assert_eq!(entry.state, CacheState::Unverified);
        assert_eq!(entry.size_bytes, Some(5));

        DigestSidecar {
            digest: "abc".into(),
            size_bytes: 5,
            source: "hub lfs metadata".into(),
        }
        .write_for(&model)
        .unwrap();
        let entry = CacheEntry::inspect(role, dir.path()).unwrap();
        assert_eq!(entry.state, CacheState::Verified);
        assert_eq!(entry.digest.as_deref(), Some("abc"));
        assert_eq!(entry.digest_source.as_deref(), Some("hub lfs metadata"));

        // A sidecar that disagrees about the size does not vouch for this file.
        DigestSidecar {
            digest: "abc".into(),
            size_bytes: 999,
            source: "hub lfs metadata".into(),
        }
        .write_for(&model)
        .unwrap();
        let entry = CacheEntry::inspect(role, dir.path()).unwrap();
        assert_eq!(entry.state, CacheState::Unverified);
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
    fn digest_mismatch_refuses_and_writes_no_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("model.gguf");
        std::fs::write(&target, b"hello").unwrap();

        let wrong = "0000000000000000000000000000000000000000000000000000000000000000";
        let err = verify_and_seal(&target, Some(wrong.to_string())).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("does not match"), "got: {message}");
        assert!(
            !DigestSidecar::path_for(&target).exists(),
            "a mismatch must not leave a sidecar behind"
        );
    }

    #[test]
    fn successful_download_writes_sidecar_with_size() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("model.gguf");
        std::fs::write(&target, b"hello").unwrap();
        let good = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

        verify_and_seal(&target, Some(good.to_string())).unwrap();

        let sidecar = DigestSidecar::read_for(&target)
            .unwrap()
            .expect("sidecar written after a matching hash");
        assert_eq!(sidecar.digest, good);
        assert_eq!(sidecar.size_bytes, 5);
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
