//! `markq embed`: fill the `embedding` column for unembedded chunks.
//!
//! Loops over batches of `embedding IS NULL` rows, embeds each text via the
//! markq-inference `Embedder` (one owner thread + bounded crossbeam channel),
//! and merge-inserts the embedded batch back keyed on `id`. The LanceDB
//! vector index is rebuilt once after the loop drains, not per batch; the
//! BM25 path is untouched.
//!
//! Ctrl-C drains cleanly: a signal flips a flag that's checked between
//! batches. An in-flight batch always finishes — never abort mid-decode.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field};
use markq_core::ChunkColumn;
use markq_index_lance::LanceIndex;
use markq_inference::{default_n_gpu_layers, ensure_model, Embedder, KnownModel};
use tracing::{info, warn};

/// How many rows to fetch / embed / merge per round-trip. Keeps memory
/// bounded on huge corpora and gives the Ctrl-C handler a frequent escape.
const EMBED_BATCH_SIZE: usize = 256;

pub struct EmbedReport {
    pub rows: u64,
    pub batches: u64,
    pub model_id: String,
    pub dim: u32,
}

pub async fn run_embed(idx: &LanceIndex) -> Result<EmbedReport> {
    let model = KnownModel::Qwen3Embedding06B;
    let model_path = ensure_model(model)
        .await
        .context("download / locate embedder model")?;

    let embedder = Embedder::load(&model_path, default_n_gpu_layers()).context("load embedder")?;
    let dim = embedder.dim();
    info!(model = model.id(), dim, "embedder ready");

    let wrote_metadata = idx
        .validate_or_record_embedder(model.id(), dim)
        .await
        .context("validate or record embedder metadata")?;
    if wrote_metadata {
        info!("recorded embedder metadata on dataset");
    }

    // Ctrl-C handler: flips a flag that we check between batches. Mid-batch
    // decode is never aborted — partial batches would just waste work.
    let stop = Arc::new(AtomicBool::new(false));
    let stop_signal = stop.clone();
    let signal_task = tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            warn!("Ctrl-C received; finishing in-flight batch then exiting");
            stop_signal.store(true, Ordering::SeqCst);
        }
    });

    let mut total_rows: u64 = 0;
    let mut total_batches: u64 = 0;
    loop {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        let batches = idx
            .scan_unembedded(Some(EMBED_BATCH_SIZE))
            .await
            .context("scan unembedded rows")?;
        if batches.is_empty() {
            break;
        }

        let mut updated = Vec::with_capacity(batches.len());
        for batch in batches {
            let new_batch = embed_batch(&embedder, batch, dim).await?;
            total_rows += new_batch.num_rows() as u64;
            updated.push(new_batch);
        }
        idx.apply_embeddings(updated)
            .await
            .context("merge embeddings into dataset")?;
        total_batches += 1;
        info!(rows = total_rows, batch = total_batches, "embedded batch");
    }

    // Don't leave the signal task hanging — abort after the loop exits
    // normally so re-running embed in the same process gets a fresh handler.
    signal_task.abort();

    // Build the HNSW vector index once, after every merge has landed. Doing
    // it per batch would trigger a full rebuild on every 256-row chunk.
    if total_rows > 0 {
        idx.rebuild_vector_index()
            .await
            .context("rebuild vector index")?;
    }

    Ok(EmbedReport {
        rows: total_rows,
        batches: total_batches,
        model_id: model.id().to_string(),
        dim,
    })
}

/// Take a RecordBatch of unembedded rows, embed each row's `text`, and
/// rebuild the same batch with the `embedding` column now populated.
async fn embed_batch(embedder: &Embedder, batch: RecordBatch, dim: u32) -> Result<RecordBatch> {
    let texts = batch
        .column_by_name(ChunkColumn::TEXT)
        .context("scanned batch missing text column")?
        .as_any()
        .downcast_ref::<StringArray>()
        .context("text column is not utf8")?;

    let n = batch.num_rows();
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n {
        let v = embedder.embed(texts.value(i).to_string()).await?;
        if v.len() as u32 != dim {
            anyhow::bail!("embedder returned dim={} but expected {dim}", v.len());
        }
        embeddings.push(v);
    }

    let embedding_arr = build_fixed_size_list(&embeddings, dim)?;
    swap_embedding_column(batch, embedding_arr)
}

/// Build a `FixedSizeList<Float32, dim>` from a row-major Vec<Vec<f32>>.
fn build_fixed_size_list(embeddings: &[Vec<f32>], dim: u32) -> Result<FixedSizeListArray> {
    let dim_usize = dim as usize;
    let mut flat = Vec::with_capacity(embeddings.len() * dim_usize);
    for v in embeddings {
        debug_assert_eq!(v.len(), dim_usize);
        flat.extend_from_slice(v);
    }
    let values = Float32Array::from(flat);
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    FixedSizeListArray::try_new(item_field, dim as i32, Arc::new(values), None)
        .context("build FixedSizeListArray for embedding column")
}

/// Replace the `embedding` column on a scanned RecordBatch with the freshly
/// computed values, preserving every other column verbatim.
fn swap_embedding_column(batch: RecordBatch, embedding: FixedSizeListArray) -> Result<RecordBatch> {
    let schema = batch.schema();
    let emb_idx = schema
        .index_of(ChunkColumn::EMBEDDING)
        .context("schema missing embedding column")?;
    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    cols[emb_idx] = Arc::new(embedding);
    RecordBatch::try_new(schema, cols).context("rebuild RecordBatch with embedding")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_fixed_size_list_flattens_row_major() {
        let embeddings = vec![
            vec![1.0_f32, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![-1.0, -2.0, -3.0, -4.0],
        ];
        let arr = build_fixed_size_list(&embeddings, 4).unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.value_length(), 4);

        // Each row is a FixedSizeList element; downcast its values to read
        // out the underlying f32 buffer and confirm row-major layout.
        let values = arr
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("FixedSizeList<Float32>");
        assert_eq!(values.len(), 3 * 4);
        let expected: Vec<f32> = embeddings.iter().flatten().copied().collect();
        let got: Vec<f32> = (0..values.len()).map(|i| values.value(i)).collect();
        assert_eq!(got, expected);
    }
}
