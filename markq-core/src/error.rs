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
