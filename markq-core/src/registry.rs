//! `~/.markq/collections.toml` registry types.
//!
//! Phase 1 ships the serde shape only — `markq collection add/list/remove`
//! lands in Phase 10 once multi-collection retrieval is wired. The registry
//! also stores the context tree (Phase 11), which is why entries carry a
//! reserved `contexts` map even though no command writes to it yet.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// One entry per registered collection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CollectionEntry {
    /// Collection name as used in `markq://<name>/...` URIs and `-c <name>`.
    pub name: String,
    /// Filesystem root indexed under this collection.
    pub root: PathBuf,
    /// Phase 11 reserved: context-tree entries keyed by URI prefix.
    /// `BTreeMap` for stable serialization order.
    #[serde(default)]
    pub contexts: BTreeMap<String, String>,
}

/// On-disk representation of `collections.toml`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Registry {
    #[serde(default)]
    pub collections: Vec<CollectionEntry>,
}

impl Registry {
    /// Default registry path: `~/.markq/collections.toml`.
    pub fn default_path() -> PathBuf {
        let mut p = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        p.push(".markq");
        p.push("collections.toml");
        p
    }

    /// Load from disk. Returns an empty registry if the file does not exist
    /// (first-run case — no collections added yet).
    pub fn load(path: &Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(s) => Self::parse(&s, path),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(Error::io(path, e)),
        }
    }

    /// Persist to disk. Creates parent dirs if missing.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let toml_str = toml::to_string_pretty(self)?;
        std::fs::write(path, toml_str).map_err(|e| Error::io(path, e))
    }

    fn parse(s: &str, path: &Path) -> Result<Self> {
        toml::from_str(s).map_err(|source| Error::ConfigParse {
            path: path.to_path_buf(),
            source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn empty_registry_round_trips() {
        let r = Registry::default();
        let s = toml::to_string_pretty(&r).unwrap();
        let r2: Registry = toml::from_str(&s).unwrap();
        assert_eq!(r, r2);
    }

    #[test]
    fn registry_with_collections_round_trips() {
        let mut contexts = BTreeMap::new();
        contexts.insert(
            "markq://notes/projects/".to_string(),
            "Active project notes".to_string(),
        );
        let r = Registry {
            collections: vec![
                CollectionEntry {
                    name: "notes".to_string(),
                    root: PathBuf::from("/home/u/notes"),
                    contexts,
                },
                CollectionEntry {
                    name: "docs".to_string(),
                    root: PathBuf::from("/home/u/docs"),
                    contexts: BTreeMap::new(),
                },
            ],
        };
        let s = toml::to_string_pretty(&r).unwrap();
        let r2: Registry = toml::from_str(&s).unwrap();
        assert_eq!(r, r2);
    }

    #[test]
    fn missing_file_returns_empty_registry() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("does-not-exist.toml");
        let r = Registry::load(&p).unwrap();
        assert!(r.collections.is_empty());
    }

    #[test]
    fn save_and_load_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("nested").join("collections.toml");
        let r = Registry {
            collections: vec![CollectionEntry {
                name: "n".to_string(),
                root: PathBuf::from("/x"),
                contexts: BTreeMap::new(),
            }],
        };
        r.save(&p).unwrap();
        let r2 = Registry::load(&p).unwrap();
        assert_eq!(r, r2);
    }

    #[test]
    fn parse_error_reports_path() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bad.toml");
        std::fs::write(&p, "not valid toml = = =").unwrap();
        let err = Registry::load(&p).unwrap_err();
        match err {
            Error::ConfigParse { path, .. } => assert_eq!(path, p),
            other => panic!("expected ConfigParse, got {other:?}"),
        }
    }
}
