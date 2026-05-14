//! Phase 3 `markq search` output formatting.
//!
//! BM25 returns `ChunkHit`s; this module turns them into one of three
//! surfaces: a human table, JSON for agents, or a deduplicated path list.

use std::io::Write;

use anyhow::Result;
use markq_core::ChunkHit;
use serde_json::json;

#[derive(Debug, Clone, Copy)]
pub enum Format {
    Table,
    Json,
    Files,
}

#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Hard cap on returned hits. `None` means "no cap" (the `--all` flag).
    /// When `None`, the caller is responsible for sizing `k` from the
    /// dataset's row count so no matches are silently dropped.
    pub top_k: Option<usize>,
    pub min_score: Option<f32>,
}

pub fn apply_filters(mut hits: Vec<ChunkHit>, opts: &SearchOptions) -> Vec<ChunkHit> {
    if let Some(min) = opts.min_score {
        hits.retain(|h| h.score >= min);
    }
    if let Some(k) = opts.top_k {
        hits.truncate(k);
    }
    hits
}

pub fn write_results<W: Write>(w: &mut W, hits: &[ChunkHit], format: Format) -> Result<()> {
    match format {
        Format::Table => write_table(w, hits),
        Format::Json => write_json(w, hits),
        Format::Files => write_files(w, hits),
    }
}

fn write_table<W: Write>(w: &mut W, hits: &[ChunkHit]) -> Result<()> {
    if hits.is_empty() {
        writeln!(w, "(no results)")?;
        return Ok(());
    }
    for (i, h) in hits.iter().enumerate() {
        writeln!(
            w,
            "{rank:>3}. {score:>7.3}  {uri}#{chunk}",
            rank = i + 1,
            score = h.score,
            uri = h.uri,
            chunk = h.chunk_index,
        )?;
        let preview = preview_line(&h.text, 100);
        writeln!(w, "     {preview}")?;
    }
    Ok(())
}

fn write_json<W: Write>(w: &mut W, hits: &[ChunkHit]) -> Result<()> {
    let arr: Vec<_> = hits
        .iter()
        .map(|h| {
            json!({
                "id": h.id,
                "path": h.path,
                "uri": h.uri,
                "chunk_index": h.chunk_index,
                "score": h.score,
                "text": h.text,
            })
        })
        .collect();
    serde_json::to_writer_pretty(&mut *w, &arr)?;
    writeln!(w)?;
    Ok(())
}

fn write_files<W: Write>(w: &mut W, hits: &[ChunkHit]) -> Result<()> {
    // Dedupe in first-seen order so the highest-scoring chunk's path wins.
    let mut seen = std::collections::HashSet::new();
    for h in hits {
        if seen.insert(h.path.clone()) {
            writeln!(w, "{}", h.path)?;
        }
    }
    Ok(())
}

fn preview_line(text: &str, max_chars: usize) -> String {
    let collapsed: String = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if collapsed.chars().count() <= max_chars {
        return collapsed;
    }
    let cut: String = collapsed.chars().take(max_chars).collect();
    format!("{cut}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(id: &str, path: &str, score: f32, text: &str) -> ChunkHit {
        ChunkHit {
            id: id.to_string(),
            path: path.to_string(),
            uri: format!("markq://default/{path}"),
            chunk_index: 0,
            text: text.to_string(),
            score,
        }
    }

    #[test]
    fn min_score_drops_below_threshold() {
        let hits = vec![h("a", "a.md", 0.9, "x"), h("b", "b.md", 0.1, "y")];
        let opts = SearchOptions {
            top_k: None,
            min_score: Some(0.5),
        };
        let out = apply_filters(hits, &opts);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, "a");
    }

    #[test]
    fn top_k_truncates_after_min_score() {
        let hits = vec![
            h("a", "a.md", 0.9, "x"),
            h("b", "b.md", 0.8, "y"),
            h("c", "c.md", 0.7, "z"),
        ];
        let opts = SearchOptions {
            top_k: Some(2),
            min_score: None,
        };
        let out = apply_filters(hits, &opts);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn files_format_dedupes_in_first_seen_order() {
        let hits = vec![
            h("a1", "a.md", 0.9, "x"),
            h("a2", "a.md", 0.8, "y"),
            h("b1", "b.md", 0.7, "z"),
        ];
        let mut buf = Vec::new();
        write_results(&mut buf, &hits, Format::Files).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(s, "a.md\nb.md\n");
    }

    #[test]
    fn json_format_is_well_formed() {
        let hits = vec![h("a", "a.md", 0.9, "hi")];
        let mut buf = Vec::new();
        write_results(&mut buf, &hits, Format::Json).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(v.as_array().unwrap().len(), 1);
        assert_eq!(v[0]["path"], "a.md");
    }

    #[test]
    fn empty_table_is_a_friendly_message() {
        let mut buf = Vec::new();
        write_results(&mut buf, &[], Format::Table).unwrap();
        assert_eq!(String::from_utf8(buf).unwrap(), "(no results)\n");
    }
}
