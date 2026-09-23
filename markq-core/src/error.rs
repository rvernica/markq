use std::path::PathBuf;

use thiserror::Error;

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Top-level markq error type. Backends wrap their own errors via `Backend`.
#[derive(Debug, Error)]
pub enum Error {
    #[error("io error at {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("config parse error in {path:?}: {source}")]
    ConfigParse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    #[error("config serialize error: {0}")]
    ConfigSerialize(#[from] toml::ser::Error),

    #[error("schema mismatch: expected schema_version={expected}, dataset has {found:?}")]
    SchemaVersionMismatch {
        expected: u32,
        found: Option<String>,
    },

    #[error(
        "embedder dimension mismatch: dataset was built with dim={dataset}, current embedder={embedder}"
    )]
    EmbedderDimMismatch { dataset: u32, embedder: u32 },

    #[error("dataset metadata missing required key: {0}")]
    MetadataMissingKey(&'static str),

    #[error("dataset metadata key {key:?} has invalid value {value:?}: {source}")]
    MetadataInvalidValue {
        key: &'static str,
        value: String,
        #[source]
        source: std::num::ParseIntError,
    },

    #[error("unknown model role {given:?}; valid roles are {valid}")]
    UnknownModelRole { given: String, valid: String },

    #[error(
        "model file {path:?} does not match its recorded digest (expected {expected}, found {found}); delete it and run `markq models pull` to fetch it again"
    )]
    ModelDigestMismatch {
        path: PathBuf,
        expected: String,
        found: String,
    },

    #[error(
        "model file {path:?} is {found} bytes, but its record says {expected}; run `markq models pull` to fetch it again"
    )]
    ModelSizeMismatch {
        path: PathBuf,
        expected: u64,
        found: u64,
    },

    #[error(
        "model file {path:?} is an unfinished download ({have} of {expected} bytes); run `markq models pull` to resume it"
    )]
    ModelFileIncomplete {
        path: PathBuf,
        have: u64,
        expected: u64,
    },

    #[error(
        "a download of the {role} model is already running (lock held on {path:?}); wait for it to finish, then try again"
    )]
    ModelDownloadInProgress { role: String, path: PathBuf },

    #[error(
        "the {role} model is not cached and {setting} forbids fetching it; run `markq models pull {role}` with network access"
    )]
    OfflineRefused { role: String, setting: String },

    #[error(
        "the {role} model is present but not verified, and {setting} forbids contacting the repository to check it; run `markq models pull {role}` with network access"
    )]
    OfflineVerificationRefused { role: String, setting: String },

    #[error("model directory {path:?} cannot be used: {reason}")]
    ModelsDirUnusable { path: PathBuf, reason: String },

    #[error("{variable} is set to {value:?}; accepted values are {accepted}")]
    InvalidOfflineValue {
        variable: &'static str,
        value: String,
        accepted: String,
    },

    #[error("backend error: {0}")]
    Backend(#[from] anyhow::Error),
}

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every new variant has to name the thing it is about, because these
    /// messages are the only repair instruction a user gets.
    #[test]
    fn error_messages_name_the_offending_subject() {
        let cases: Vec<(Error, &str)> = vec![
            (
                Error::UnknownModelRole {
                    given: "embeder".into(),
                    valid: "embedder, reranker".into(),
                },
                "embeder",
            ),
            (
                Error::ModelDigestMismatch {
                    path: "/cache/model.gguf".into(),
                    expected: "abc".into(),
                    found: "def".into(),
                },
                "/cache/model.gguf",
            ),
            (
                Error::ModelSizeMismatch {
                    path: "/cache/model.gguf".into(),
                    expected: 10,
                    found: 4,
                },
                "/cache/model.gguf",
            ),
            (
                Error::ModelFileIncomplete {
                    path: "/cache/model.gguf".into(),
                    have: 4,
                    expected: 10,
                },
                "/cache/model.gguf",
            ),
            (
                Error::ModelDownloadInProgress {
                    role: "reranker".into(),
                    path: "/cache/model.gguf.lock".into(),
                },
                "reranker",
            ),
            (
                Error::OfflineRefused {
                    role: "embedder".into(),
                    setting: "--offline".into(),
                },
                "--offline",
            ),
            (
                Error::ModelsDirUnusable {
                    path: "/nope".into(),
                    reason: "no such directory".into(),
                },
                "/nope",
            ),
            (
                Error::InvalidOfflineValue {
                    variable: "MARKQ_OFFLINE",
                    value: "maybe".into(),
                    accepted: "1, true, 0, false".into(),
                },
                "maybe",
            ),
        ];

        for (error, subject) in cases {
            let rendered = error.to_string();
            assert!(
                rendered.contains(subject),
                "{rendered:?} does not name {subject:?}"
            );
        }
    }

    /// Committed output describes what a condition is, never where it sits on
    /// an internal roadmap.
    #[test]
    fn messages_are_free_of_roadmap_language() {
        let rendered = [
            Error::UnknownModelRole {
                given: "x".into(),
                valid: "embedder, reranker".into(),
            },
            Error::ModelDigestMismatch {
                path: "/m".into(),
                expected: "a".into(),
                found: "b".into(),
            },
            Error::ModelSizeMismatch {
                path: "/m".into(),
                expected: 1,
                found: 2,
            },
            Error::ModelFileIncomplete {
                path: "/m".into(),
                have: 1,
                expected: 2,
            },
            Error::ModelDownloadInProgress {
                role: "embedder".into(),
                path: "/m.lock".into(),
            },
            Error::OfflineRefused {
                role: "embedder".into(),
                setting: "MARKQ_OFFLINE".into(),
            },
            Error::ModelsDirUnusable {
                path: "/m".into(),
                reason: "r".into(),
            },
            Error::InvalidOfflineValue {
                variable: "MARKQ_OFFLINE",
                value: "maybe".into(),
                accepted: "1, true".into(),
            },
        ]
        .map(|error| error.to_string().to_ascii_lowercase());

        for message in rendered {
            assert!(
                !message.contains("phase"),
                "roadmap language in {message:?}"
            );
            assert!(!message.contains("qmd/"), "qmd citation in {message:?}");
        }
    }
}
