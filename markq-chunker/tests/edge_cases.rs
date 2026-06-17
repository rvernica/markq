//! Edge-case tests for the chunker.
//!
//! These run against synthetic fixtures embedded in the test source rather
//! than disk fixtures. The `corpus/` directory will be wired in as a separate
//! evaluation harness later (per `spikes/PHASE1_FOLLOWUPS.md`).

use markq_chunker::{chunk_markdown, ApproxTokenizer, ChunkOptions};

fn opts_small() -> ChunkOptions {
    // Small budget so the packer actually has to split on multi-paragraph
    // fixtures without needing kilobytes of input.
    ChunkOptions {
        max_tokens: 60,
        overlap_tokens: 12,
        min_tokens: 8,
    }
}

/// Property: every byte of the source is covered by at least one chunk's
/// `[start_byte, end_byte)` range. With overlap, a byte may appear in two
/// adjacent chunks; the contract is "no gaps", not "no duplicates".
fn assert_full_coverage(src: &str, chunks: &[markq_chunker::Chunk]) {
    if src.is_empty() {
        assert!(chunks.is_empty(), "empty input must yield no chunks");
        return;
    }
    assert!(!chunks.is_empty(), "non-empty input must yield ≥1 chunk");

    // Sort by start; assert chunk[0] starts at 0, chunk[n-1] ends at src.len(),
    // and each consecutive pair has start[i+1] <= end[i] (no gap).
    let mut sorted = chunks.to_vec();
    sorted.sort_by_key(|c| c.start_byte);
    assert_eq!(sorted[0].start_byte, 0, "first chunk must start at byte 0");
    assert_eq!(
        sorted.last().unwrap().end_byte,
        src.len(),
        "last chunk must end at src.len() ({} chunks: {:?})",
        sorted.len(),
        sorted
            .iter()
            .map(|c| (c.start_byte, c.end_byte))
            .collect::<Vec<_>>()
    );
    for w in sorted.windows(2) {
        assert!(
            w[1].start_byte <= w[0].end_byte,
            "gap between chunk ending at {} and chunk starting at {}",
            w[0].end_byte,
            w[1].start_byte
        );
    }

    // And: each chunk's `text` must equal the slice it claims.
    for c in chunks {
        assert_eq!(
            c.text,
            &src[c.start_byte..c.end_byte],
            "chunk {} text drifted from its byte range",
            c.index
        );
    }
}

#[test]
fn yaml_frontmatter_is_attached_to_chunk_zero() {
    let src = "---\ntitle: Doc\ntags: [a, b]\n---\n# Heading\n\nBody paragraph.\n";
    let out = chunk_markdown(src, &ChunkOptions::default(), &ApproxTokenizer);
    assert_eq!(out.len(), 1);
    assert!(out[0].text.starts_with("---\ntitle: Doc"));
    assert!(out[0].text.contains("# Heading"));
}

#[test]
fn nested_fences_in_a_list_stay_atomic() {
    // The inner fence (4 backticks containing a 3-backtick block) is a
    // classic "split me wrong" trap: a naive scanner that treats every "```"
    // as a delimiter would emit a code chunk in the middle of the list.
    let src = "\
- intro item

  ````
  ```
  inner ``` fence
  ```
  ````

- after
";
    let out = chunk_markdown(src, &ChunkOptions::default(), &ApproxTokenizer);
    assert_eq!(out.len(), 1, "nested fence should not force a split");
    assert!(out[0].text.contains("inner ``` fence"));
    assert_full_coverage(src, &out);
}

#[test]
fn gfm_table_is_not_split_mid_row_under_pressure() {
    // Drive the packer with a tight budget; the table must remain atomic
    // (one chunk's bytes fully contain the table) even when emitting it
    // alone exceeds max_tokens.
    let src = "\
Intro paragraph that gives the packer something to chew on before the table.

| col a | col b | col c |
|-------|-------|-------|
| r1a   | r1b   | r1c   |
| r2a   | r2b   | r2c   |
| r3a   | r3b   | r3c   |
| r4a   | r4b   | r4c   |

Trailing paragraph that follows the table.
";
    let out = chunk_markdown(src, &opts_small(), &ApproxTokenizer);
    assert_full_coverage(src, &out);
    // Find the chunk that starts the table row line and assert the closing
    // row is in the same chunk.
    let table_marker = "| col a | col b | col c |";
    let last_row = "| r4a   | r4b   | r4c   |";
    let owner = out
        .iter()
        .find(|c| c.text.contains(table_marker))
        .expect("a chunk must contain the table header");
    assert!(
        owner.text.contains(last_row),
        "table was split mid-rows across chunks"
    );
}

#[test]
fn file_without_trailing_newline_round_trips() {
    let src = "# Title\n\nbody without newline";
    let out = chunk_markdown(src, &ChunkOptions::default(), &ApproxTokenizer);
    assert_full_coverage(src, &out);
    assert_eq!(out.last().unwrap().end_byte, src.len());
}

#[test]
fn code_only_file_emits_one_chunk_covering_everything() {
    let src = "```python\nx = 1\nprint(x)\n```\n";
    let out = chunk_markdown(src, &ChunkOptions::default(), &ApproxTokenizer);
    assert_eq!(out.len(), 1);
    assert_full_coverage(src, &out);
}

#[test]
fn empty_file_yields_no_chunks() {
    let out = chunk_markdown("", &ChunkOptions::default(), &ApproxTokenizer);
    assert!(out.is_empty());
}

#[test]
fn whitespace_only_file_is_one_chunk() {
    let src = "\n\n   \n\n";
    let out = chunk_markdown(src, &ChunkOptions::default(), &ApproxTokenizer);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].text, src);
}

#[test]
fn embedded_html_block_is_atomic() {
    let src = "\
Lead paragraph.

<div class=\"note\">
  <p>HTML body that must not be split.</p>
  <ul><li>one</li><li>two</li></ul>
</div>

Tail paragraph.
";
    let out = chunk_markdown(src, &opts_small(), &ApproxTokenizer);
    assert_full_coverage(src, &out);
    let owner = out
        .iter()
        .find(|c| c.text.contains("<div class=\"note\">"))
        .expect("html block must live in some chunk");
    assert!(owner.text.contains("</div>"), "html block was split");
}

#[test]
fn long_doc_splits_at_heading_boundaries_when_possible() {
    // Build a doc with several headed sections, each ~30-40 approx tokens.
    // With max_tokens=60 the packer should close near heading boundaries.
    let mut src = String::new();
    for i in 0..6 {
        src.push_str(&format!(
            "# Section {i}\n\nThis is paragraph {i} with a moderate amount of body text \
            so the packer has tokens to count against the budget.\n\n"
        ));
    }
    let out = chunk_markdown(&src, &opts_small(), &ApproxTokenizer);
    assert_full_coverage(&src, &out);
    assert!(
        out.len() >= 2,
        "expected the budget to force at least one split; got {} chunks",
        out.len()
    );
    // Each non-first chunk should contain a heading marker — i.e. the
    // packer closed the previous chunk on a heading boundary, even though
    // the new chunk's overlap region rewinds into the prior paragraph.
    for c in out.iter().skip(1) {
        assert!(
            c.text.contains("# Section "),
            "non-first chunk {} missing a `# Section` heading: {:?}",
            c.index,
            &c.text
        );
    }
}

#[test]
fn zero_overlap_still_covers_inter_block_whitespace() {
    // Regression: with `overlap_tokens = 0` (or when overlap_seed declines
    // to rewind across a heading boundary), the next chunk would start at
    // raw_blocks[i].start while the previous chunk ended at
    // raw_blocks[i-1].end. Any inter-block whitespace between those two
    // offsets used to become an uncovered gap, violating assert_full_coverage.
    let mut src = String::new();
    for i in 0..6 {
        src.push_str(&format!(
            "# Section {i}\n\nParagraph {i} with enough body text to push the \
            packer over its tight budget and force a split on the next heading.\n\n"
        ));
    }
    let opts = ChunkOptions {
        max_tokens: 60,
        overlap_tokens: 0,
        min_tokens: 8,
    };
    let out = chunk_markdown(&src, &opts, &ApproxTokenizer);
    assert!(out.len() >= 2, "tight budget should force a split");
    assert_full_coverage(&src, &out);
}

#[test]
fn deterministic_output_for_same_input() {
    let src = "# A\n\npara one\n\n# B\n\npara two\n\n# C\n\npara three\n";
    let opts = ChunkOptions::default();
    let a = chunk_markdown(src, &opts, &ApproxTokenizer);
    let b = chunk_markdown(src, &opts, &ApproxTokenizer);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.start_byte, y.start_byte);
        assert_eq!(x.end_byte, y.end_byte);
        assert_eq!(x.token_count, y.token_count);
    }
}
