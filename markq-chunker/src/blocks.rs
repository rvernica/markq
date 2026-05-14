//! Top-level block segmentation.
//!
//! Walks `pulldown-cmark`'s offset iterator and emits the byte range of each
//! depth-0 block (paragraph, heading, fenced code block, list, blockquote,
//! HTML block, table, thematic break, footnote definition). Inline events
//! and nested blocks (list items, table rows, nested quotes, nested fenced
//! code inside a list) are absorbed into their enclosing top-level block —
//! we never split mid-block.
//!
//! For headings we also record the level so the chunk packer can prefer
//! splitting on heading boundaries.

use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    Heading(u8),
    Paragraph,
    /// Fenced or indented code block. Treated as atomic — never split.
    Code,
    /// Ordered or unordered list. Atomic for v1; large lists become large
    /// chunks. Splitting at item boundaries is a later refinement.
    List,
    BlockQuote,
    /// GFM table. Atomic.
    Table,
    /// Raw HTML block. Atomic.
    Html,
    /// Thematic break (`---`). Cheap natural break point.
    Rule,
    /// Footnote definition. Atomic.
    FootnoteDef,
    /// Catch-all for anything new pulldown-cmark adds.
    Other,
}

impl BlockKind {
    /// Heading boundaries are the preferred chunk split point. Code/Table/
    /// HTML blocks must not be split mid-block; the packer treats them as a
    /// single unit even when oversize.
    pub fn is_heading(self) -> bool {
        matches!(self, BlockKind::Heading(_))
    }

    /// Blocks the packer must keep whole. When `push_block` coalesces an
    /// atomic block into a non-atomic predecessor, the merged span is
    /// promoted to the atomic kind so the packer still treats it as a unit.
    pub fn is_atomic(self) -> bool {
        matches!(
            self,
            BlockKind::Code | BlockKind::Table | BlockKind::Html | BlockKind::FootnoteDef
        )
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    pub kind: BlockKind,
    /// Half-open byte range into the source (or the post-frontmatter slice;
    /// the caller decides which slice it parses).
    pub start: usize,
    pub end: usize,
}

/// Parse `src` and return its top-level blocks in source order.
///
/// `src` should already have any YAML frontmatter stripped (see
/// [`crate::frontmatter`]); offsets returned here are relative to `src`.
pub fn segment(src: &str) -> Vec<Block> {
    if src.is_empty() {
        return Vec::new();
    }

    let opts = Options::ENABLE_TABLES
        | Options::ENABLE_FOOTNOTES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_HEADING_ATTRIBUTES;
    let parser = Parser::new_ext(src, opts).into_offset_iter();

    let mut blocks: Vec<Block> = Vec::new();
    let mut depth: i32 = 0;
    let mut current: Option<(BlockKind, usize)> = None;

    for (event, range) in parser {
        match event {
            Event::Start(tag) => {
                if depth == 0 {
                    let kind = top_level_kind(&tag);
                    current = Some((kind, range.start));
                }
                depth += 1;
            }
            Event::End(_) => {
                depth -= 1;
                if depth == 0 {
                    if let Some((kind, start)) = current.take() {
                        // Trust the End event's range.end — it points at the
                        // byte after the block's terminator (newline included).
                        push_block(&mut blocks, src, kind, start, range.end);
                    }
                }
            }
            Event::Rule if depth == 0 => {
                push_block(&mut blocks, src, BlockKind::Rule, range.start, range.end);
            }
            Event::Html(_) if depth == 0 => {
                push_block(&mut blocks, src, BlockKind::Html, range.start, range.end);
            }
            // Text / Code / SoftBreak / HardBreak / etc. only matter inside a
            // block (depth > 0); their bytes are already covered by the
            // enclosing Start..End range.
            _ => {}
        }
    }

    blocks
}

fn top_level_kind(tag: &Tag) -> BlockKind {
    match tag {
        Tag::Heading { level, .. } => BlockKind::Heading(heading_level_to_u8(*level)),
        Tag::Paragraph => BlockKind::Paragraph,
        Tag::CodeBlock(_) => BlockKind::Code,
        Tag::List(_) => BlockKind::List,
        Tag::BlockQuote(_) => BlockKind::BlockQuote,
        Tag::Table(_) => BlockKind::Table,
        Tag::HtmlBlock => BlockKind::Html,
        Tag::FootnoteDefinition(_) => BlockKind::FootnoteDef,
        _ => BlockKind::Other,
    }
}

fn heading_level_to_u8(l: HeadingLevel) -> u8 {
    match l {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

fn push_block(out: &mut Vec<Block>, src: &str, kind: BlockKind, start: usize, end: usize) {
    let end = end.min(src.len());
    if end <= start {
        return;
    }
    // Coalesce adjacent overlap from pulldown-cmark events whose ranges
    // touch or overlap the previous block (e.g. an Html event emitted
    // immediately after a paragraph that already covered the same byte).
    if let Some(last) = out.last_mut() {
        if start < last.end {
            // Overlapping — extend if the new event reaches further.
            if end > last.end {
                last.end = end;
            }
            // If the incoming kind is atomic and the existing one is not,
            // promote: the packer relies on `is_atomic` to keep code /
            // table / HTML / footnote definitions whole, so a paragraph
            // that swallows an Html event must report as Html.
            if kind.is_atomic() && !last.kind.is_atomic() {
                last.kind = kind;
            }
            return;
        }
    }
    out.push(Block { kind, start, end });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_source_has_no_blocks() {
        assert!(segment("").is_empty());
    }

    #[test]
    fn heading_paragraph_split() {
        let src = "# Title\n\nA paragraph.\n";
        let blocks = segment(src);
        let kinds: Vec<_> = blocks.iter().map(|b| b.kind).collect();
        assert_eq!(kinds, vec![BlockKind::Heading(1), BlockKind::Paragraph]);
    }

    #[test]
    fn fenced_code_is_one_block() {
        let src = "Intro.\n\n```rust\nfn x() {}\n```\n\nOutro.\n";
        let blocks = segment(src);
        let kinds: Vec<_> = blocks.iter().map(|b| b.kind).collect();
        assert_eq!(
            kinds,
            vec![BlockKind::Paragraph, BlockKind::Code, BlockKind::Paragraph]
        );
        // The code block must contain its opening and closing fences.
        let code = blocks.iter().find(|b| b.kind == BlockKind::Code).unwrap();
        let body = &src[code.start..code.end];
        assert!(body.contains("```rust"));
        assert!(body.trim_end().ends_with("```"));
    }

    #[test]
    fn nested_fence_inside_list_stays_one_list_block() {
        // A code fence inside a list item — pulldown-cmark nests it; the
        // top-level block is the list, and we must not emit the inner code
        // as its own top-level block.
        let src = "- item\n\n  ```\n  inner fence\n  ```\n\n- two\n";
        let blocks = segment(src);
        let kinds: Vec<_> = blocks.iter().map(|b| b.kind).collect();
        assert_eq!(kinds, vec![BlockKind::List]);
    }

    #[test]
    fn gfm_table_is_one_block() {
        let src = "| a | b |\n|---|---|\n| 1 | 2 |\n";
        let blocks = segment(src);
        let kinds: Vec<_> = blocks.iter().map(|b| b.kind).collect();
        assert_eq!(kinds, vec![BlockKind::Table]);
    }
}
