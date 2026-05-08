//! YAML frontmatter detection.
//!
//! `pulldown-cmark` does not strip `---`-fenced YAML frontmatter; if we hand
//! it the raw source, the leading `---` parses as a thematic break and the
//! YAML body becomes a paragraph. We strip frontmatter ourselves and return
//! the byte range so the caller can either preserve it on the first chunk
//! or drop it.

/// Byte range of a leading YAML frontmatter block, or `None` if absent.
///
/// Recognized form (CommonMark-friendly): the file starts with `---` followed
/// by a newline, and a later line that is exactly `---` (optionally trailed
/// by `\r`) terminates the block. The returned range is the half-open span
/// `[0, end)` where `end` points just past the trailing newline of the
/// closing `---` line — i.e. body chunking should start at `end`.
pub fn detect(src: &str) -> Option<std::ops::Range<usize>> {
    let bytes = src.as_bytes();
    // Opening: `---` then `\n` or `\r\n` at byte 0.
    if !bytes.starts_with(b"---") {
        return None;
    }
    let after_dashes = 3;
    let after_open_newline = match bytes.get(after_dashes) {
        Some(b'\n') => after_dashes + 1,
        Some(b'\r') if bytes.get(after_dashes + 1) == Some(&b'\n') => after_dashes + 2,
        _ => return None,
    };

    // Walk lines looking for a closing `---` line.
    let mut line_start = after_open_newline;
    while line_start < bytes.len() {
        let nl = bytes[line_start..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|p| line_start + p);
        let line_end = nl.unwrap_or(bytes.len());
        let line = &bytes[line_start..line_end];
        let trimmed = if line.last() == Some(&b'\r') {
            &line[..line.len() - 1]
        } else {
            line
        };
        if trimmed == b"---" {
            // Include the trailing newline (if any) in the stripped range so
            // the body slice begins on a fresh line.
            let end = nl.map(|p| p + 1).unwrap_or(bytes.len());
            return Some(0..end);
        }
        match nl {
            Some(p) => line_start = p + 1,
            None => return None,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_simple_frontmatter() {
        let src = "---\ntitle: Hi\n---\n# Body\n";
        let r = detect(src).expect("frontmatter");
        assert_eq!(&src[r.clone()], "---\ntitle: Hi\n---\n");
        assert_eq!(&src[r.end..], "# Body\n");
    }

    #[test]
    fn no_frontmatter_when_no_opener() {
        assert!(detect("# Title\n---\nbody\n").is_none());
    }

    #[test]
    fn unterminated_frontmatter_is_none() {
        assert!(detect("---\ntitle: oops\nbody\n").is_none());
    }

    #[test]
    fn crlf_frontmatter() {
        let src = "---\r\nk: v\r\n---\r\nbody\r\n";
        let r = detect(src).expect("frontmatter");
        assert_eq!(&src[r.end..], "body\r\n");
    }
}
