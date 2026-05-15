//! Embed a single query string and print the vector. Lets external tools
//! (DuckDB's `lance_vector_search`, Python via pylance, jq pipelines)
//! run their own vector search against the markq dataset without
//! needing to load a GGUF themselves.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use markq_core::Index;
use markq_index_lance::LanceIndex;
use markq_inference::{default_n_gpu_layers, ensure_model, Embedder, KnownModel};

/// Embed `query` with the canonical embedder and write the vector as a
/// single-line JSON array to `out`. If `dataset_path` points at an
/// existing dataset that already records an `embedder_model`, the model
/// IDs must match — otherwise the printed vector would not be cosine-
/// compatible with the dataset's stored vectors.
pub async fn run_embed_query<W: Write>(
    dataset_path: &Path,
    query: &str,
    out: &mut W,
) -> Result<()> {
    let model = KnownModel::Qwen3Embedding06B;

    if dataset_path.exists() {
        let idx = LanceIndex::open(dataset_path)
            .await
            .context("open dataset")?;
        let md = idx.metadata().await.context("read dataset metadata")?;
        match md.embedder_model.as_deref() {
            Some(existing) if existing != model.id() => {
                anyhow::bail!(
                    "dataset was built with embedder {existing}, but this build only knows {}",
                    model.id()
                );
            }
            Some(_) => {}
            None => {
                // A printed vector would not be useful against a dataset
                // with no stored vectors; match `markq vsearch`'s guard so
                // the failure mode is the same across read paths.
                anyhow::bail!(
                    "no embeddings in this dataset; run `markq embed` first to populate them"
                );
            }
        }
    }

    let model_path = ensure_model(model).await.context("locate embedder model")?;
    let embedder =
        Embedder::load(&model_path, default_n_gpu_layers()).context("load embedder")?;
    let vec = embedder
        .embed(query.to_string())
        .await
        .context("embed query")?;

    write_json_array(out, &vec)
}

/// Write `v` as a single-line JSON array on `out` (newline-terminated).
/// Rejects non-finite components (NaN, ±inf) up front — serde_json would
/// silently emit `null` for those, which a downstream `FLOAT[N]` cast in
/// DuckDB / pylance would either reject or treat as zero.
pub fn write_json_array<W: Write>(out: &mut W, v: &[f32]) -> Result<()> {
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        anyhow::bail!(
            "embedding contains non-finite value at index {i} ({}); refusing to emit invalid JSON",
            v[i],
        );
    }
    serde_json::to_writer(&mut *out, v).context("serialize embedding as JSON")?;
    writeln!(out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn write_json_array_round_trips_finite_floats() {
        let v: Vec<f32> = vec![0.0, -1.5, 3.4e-10, 1.0];
        let mut buf: Vec<u8> = Vec::new();
        write_json_array(&mut buf, &v).expect("serialize");
        let parsed: Vec<f32> =
            serde_json::from_slice(&buf).expect("parses back as a JSON float array");
        assert_eq!(parsed, v);
        assert_eq!(buf.last(), Some(&b'\n'));
    }

    #[test]
    fn write_json_array_rejects_non_finite() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let v: Vec<f32> = vec![0.0, bad, 1.0];
            let mut buf: Vec<u8> = Vec::new();
            let res = write_json_array(&mut buf, &v);
            assert!(res.is_err(), "non-finite {bad} must error");
            let msg = format!("{:#}", res.unwrap_err());
            assert!(
                msg.contains("non-finite value at index 1"),
                "expected index-1 message, got: {msg}",
            );
            assert!(
                buf.is_empty(),
                "must not write partial JSON for non-finite input ({bad})",
            );
        }
    }

    #[tokio::test]
    async fn bails_when_dataset_has_no_embeddings() {
        let dir = TempDir::new().unwrap();
        let dataset = dir.path().join("chunks.lance");
        // open_or_create writes the markq.* metadata keys but leaves
        // embedder_model / embedder_dim unset until run_embed is called.
        let _ = LanceIndex::open_or_create(&dataset)
            .await
            .expect("create empty dataset");

        let mut buf: Vec<u8> = Vec::new();
        let err = run_embed_query(&dataset, "anything", &mut buf)
            .await
            .expect_err("empty dataset must error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no embeddings in this dataset"),
            "expected no-embeddings bail, got: {msg}",
        );
        assert!(buf.is_empty(), "must not write a partial vector on failure");
    }

    #[tokio::test]
    async fn bails_on_embedder_model_mismatch() {
        let dir = TempDir::new().unwrap();
        let dataset = dir.path().join("chunks.lance");
        let idx = LanceIndex::open_or_create(&dataset)
            .await
            .expect("create dataset");
        // Record a fake embedder model so the validation branch fires
        // before any GGUF load is attempted.
        idx.validate_or_record_embedder("fake/other-embedder", 1024)
            .await
            .expect("record fake embedder");

        let mut buf: Vec<u8> = Vec::new();
        let err = run_embed_query(&dataset, "anything", &mut buf)
            .await
            .expect_err("mismatch must error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("fake/other-embedder"),
            "expected recorded embedder id in message, got: {msg}",
        );
        assert!(
            msg.contains(KnownModel::Qwen3Embedding06B.id()),
            "expected known embedder id in message, got: {msg}",
        );
        assert!(buf.is_empty(), "must not write a partial vector on failure");
    }
}
