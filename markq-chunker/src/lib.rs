//! markdown chunker.
//!
//! Splits a markdown source into roughly fixed-token chunks while respecting
//! structural boundaries: headings are preferred split points, fenced code
//! blocks / tables / HTML blocks are atomic (never split mid-block), and
//! consecutive chunks share ~15% token overlap so retrieval doesn't lose
//! context across a hard boundary.
//!
//! The chunker is pure-CPU and inference-free; tokens are counted via the
//! pluggable [`Tokenize`] trait so can swap the approximator for the
//! real Qwen embedder tokenizer without touching this crate.

pub mod blocks;
pub mod frontmatter;
pub mod tokenize;

use blocks::{segment, Block};

pub use blocks::BlockKind;

pub use tokenize::{ApproxTokenizer, Tokenize};

/// Chunker parameters. Defaults track `plan.md`: ~900-token target, 15% overlap.
#[derive(Debug, Clone, Copy)]
pub struct ChunkOptions {
    /// Soft maximum tokens per chunk. Atomic blocks (code, table, HTML) may
    /// individually exceed this — they are emitted as a single oversize chunk
    /// rather than being split mid-block.
    pub max_tokens: usize,
    /// Approximate token overlap between consecutive chunks. The packer
    /// rewinds by whole blocks until the rewound suffix reaches this many
    /// tokens (or the chunk start, whichever comes first).
    pub overlap_tokens: usize,
    /// Minimum tokens before emitting a chunk. Avoids degenerate one-block
    /// chunks when a heading sits next to a tiny paragraph.
    pub min_tokens: usize,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            max_tokens: 900,
            overlap_tokens: 135, // ~15% of 900
            min_tokens: 64,
        }
    }
}

/// One chunk of the source. Byte ranges are relative to the *original* source
/// passed to [`chunk_markdown`] (frontmatter offsets included), so a caller
/// can slice the source directly to recover `text`.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub index: usize,
    pub start_byte: usize,
    pub end_byte: usize,
    pub text: String,
    pub token_count: usize,
}

/// Chunk a markdown source.
///
/// Algorithm:
/// 1. Strip leading YAML frontmatter (preserved on chunk 0 as a preamble so
///    titles/tags don't get lost).
/// 2. Segment the body into top-level blocks via pulldown-cmark.
/// 3. Greedy-pack blocks into chunks under `max_tokens`, preferring to close
///    a chunk just before a heading once we've crossed `min_tokens`.
/// 4. Seed each new chunk with `overlap_tokens` of trailing context from the
///    previous chunk's blocks.
pub fn chunk_markdown(src: &str, opts: &ChunkOptions, tok: &dyn Tokenize) -> Vec<Chunk> {
    if src.is_empty() {
        return Vec::new();
    }

    let fm = frontmatter::detect(src);
    let body_start = fm.as_ref().map(|r| r.end).unwrap_or(0);
    let body = &src[body_start..];

    let mut raw_blocks = segment(body);
    // Re-base block offsets onto the original source so callers can slice
    // `src[chunk.start_byte..chunk.end_byte]` without knowing about
    // frontmatter stripping.
    for b in &mut raw_blocks {
        b.start += body_start;
        b.end += body_start;
    }

    if raw_blocks.is_empty() {
        // Code-only / whitespace-only / frontmatter-only: emit one chunk
        // covering the entire source so byte coverage holds.
        let text = src.to_string();
        let token_count = tok.count_tokens(&text);
        return vec![Chunk {
            index: 0,
            start_byte: 0,
            end_byte: src.len(),
            text,
            token_count,
        }];
    }

    // Pre-compute per-block token counts on the original-source slice.
    let block_tokens: Vec<usize> = raw_blocks
        .iter()
        .map(|b| tok.count_tokens(&src[b.start..b.end]))
        .collect();

    // Pack greedily.
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut i = 0usize;
    let n = raw_blocks.len();
    let preamble_end = body_start;

    while i < n {
        // Decide overlap seed: walk backwards from `i` over the previous
        // chunk's blocks until we accumulate `overlap_tokens`. Overlap only
        // applies after the first chunk.
        let overlap_start_block = if chunks.is_empty() {
            i
        } else {
            overlap_seed(&raw_blocks, &block_tokens, i, opts.overlap_tokens)
        };

        let chunk_start_byte = if chunks.is_empty() {
            // First chunk: include any preamble (frontmatter + leading
            // whitespace before the first block) so byte 0 is covered.
            0
        } else {
            raw_blocks[overlap_start_block].start
        };

        // Accumulate tokens from `overlap_start_block` through some `j > i`,
        // closing the chunk when adding the next block would exceed the
        // budget AND we've already cleared `min_tokens`. Atomic-but-oversize
        // blocks (e.g. a giant code fence) are emitted alone.
        let mut acc_tokens: usize = (overlap_start_block..i).map(|k| block_tokens[k]).sum();
        let mut j = i;

        while j < n {
            let next = block_tokens[j];

            let would_exceed = acc_tokens + next > opts.max_tokens;
            let already_have_content = j > i;

            if would_exceed && already_have_content && acc_tokens >= opts.min_tokens {
                break;
            }

            // Prefer closing on a heading boundary once we have enough.
            if already_have_content
                && raw_blocks[j].kind.is_heading()
                && acc_tokens >= opts.min_tokens
            {
                break;
            }

            acc_tokens += next;
            j += 1;

            if acc_tokens >= opts.max_tokens && j < n {
                // Hit the budget exactly — close here unless the next block
                // is part of an atomic continuation we've already absorbed.
                break;
            }
        }

        // Always make at least one block of forward progress so we don't
        // loop forever on a single oversize block.
        if j == i {
            j = i + 1;
            acc_tokens += block_tokens[i];
        }

        let end_byte = raw_blocks[j - 1].end;
        // For chunks after the first, ensure start <= end (overlap_seed can
        // legitimately point at `i` itself when no overlap fits).
        let start_byte = chunk_start_byte.min(end_byte);
        // Tail of the last chunk should swallow trailing whitespace so byte
        // coverage holds without an empty trailing chunk.
        let end_byte = if j == n { src.len() } else { end_byte };
        // First chunk always starts at 0 to absorb any pre-block whitespace
        // / frontmatter; we already set chunk_start_byte = 0 above.
        let _ = preamble_end;

        let text = src[start_byte..end_byte].to_string();
        let token_count = tok.count_tokens(&text);
        chunks.push(Chunk {
            index: chunks.len(),
            start_byte,
            end_byte,
            text,
            token_count,
        });

        i = j;
    }

    chunks
}

/// Walk backwards from `current_block` accumulating tokens until we reach
/// `target_overlap` (or the start of the previous chunk's region). Returns
/// the block index that should seed the next chunk.
fn overlap_seed(
    blocks: &[Block],
    block_tokens: &[usize],
    current_block: usize,
    target_overlap: usize,
) -> usize {
    if target_overlap == 0 || current_block == 0 {
        return current_block;
    }
    let mut acc = 0usize;
    let mut k = current_block;
    while k > 0 {
        let prev = k - 1;
        // Don't pull a heading into the next chunk's overlap — the heading is
        // what we just split on; repeating it inflates BM25 weight unfairly.
        if blocks[prev].kind.is_heading() {
            break;
        }
        let next_acc = acc + block_tokens[prev];
        if next_acc > target_overlap && acc > 0 {
            break;
        }
        acc = next_acc;
        k = prev;
        if acc >= target_overlap {
            break;
        }
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_no_chunks() {
        let out = chunk_markdown("", &ChunkOptions::default(), &ApproxTokenizer);
        assert!(out.is_empty());
    }

    #[test]
    fn small_doc_is_a_single_chunk() {
        let src = "# Title\n\nHello world.\n";
        let out = chunk_markdown(src, &ChunkOptions::default(), &ApproxTokenizer);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].start_byte, 0);
        assert_eq!(out[0].end_byte, src.len());
    }

    #[test]
    fn frontmatter_is_preserved_on_first_chunk() {
        let src = "---\ntitle: A\n---\n# Body\n\nHi.\n";
        let out = chunk_markdown(src, &ChunkOptions::default(), &ApproxTokenizer);
        assert!(out[0].text.starts_with("---\ntitle: A\n---"));
    }

    #[test]
    fn code_only_file_is_one_chunk() {
        let src = "```\njust code\nno prose\n```\n";
        let out = chunk_markdown(src, &ChunkOptions::default(), &ApproxTokenizer);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, src);
    }

    #[test]
    fn no_trailing_newline_still_chunks() {
        let src = "# Title\n\nbody";
        let out = chunk_markdown(src, &ChunkOptions::default(), &ApproxTokenizer);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].end_byte, src.len());
    }
}
