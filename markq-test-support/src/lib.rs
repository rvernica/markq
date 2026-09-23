//! A local stand-in for the Hugging Face hub, so download tests never touch the
//! network. `HFClient::builder().endpoint(..)` points the real client at it.
//!
//! It answers the three requests the download path makes: a `HEAD` for file
//! metadata, a `GET` for content that honors `Range`, and a tree listing that
//! carries the LFS digest. Requests are counted so a test can assert that no
//! request was made at all.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// The most recent content request's byte range, shared with the connection
/// handlers so a test can read what was actually asked for.
type RangeLog = Arc<Mutex<Option<(u64, Option<u64>)>>>;

/// How the hub should behave for content requests, so a test can drive the
/// failure paths without racing a real connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Behavior {
    /// Serve the requested bytes in full.
    Complete,
    /// Send only the first `n` bytes of the body, then close the connection.
    TruncateAfter(usize),
    /// Answer every content request with 404.
    NotFound,
    /// Omit the LFS block from the tree listing, so no digest is declared.
    NoDeclaredDigest,
    /// Declare a digest that does not match the content actually served.
    WrongDeclaredDigest,
    /// Answer a ranged request with the whole file, as a proxy or mirror that
    /// does not implement ranges would.
    IgnoreRange,
}

pub struct FakeHub {
    endpoint: String,
    content: Vec<u8>,
    sha256: String,
    filename: String,
    requests: Arc<AtomicUsize>,
    last_range: RangeLog,
}

impl FakeHub {
    /// Start a hub serving `content` for every file, on a port the OS picks.
    pub async fn spawn(content: Vec<u8>) -> Self {
        Self::spawn_with(content, Behavior::Complete).await
    }

    pub async fn spawn_with(content: Vec<u8>, behavior: Behavior) -> Self {
        Self::spawn_for("model.gguf", content, behavior).await
    }

    /// Start a hub whose tree listing advertises `filename`, so a caller that
    /// looks the file up by name finds it.
    pub async fn spawn_for(filename: &str, content: Vec<u8>, behavior: Behavior) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        let port = listener.local_addr().expect("local addr").port();
        let sha256 = hex_digest(&content);
        let requests = Arc::new(AtomicUsize::new(0));

        let body = content.clone();
        let digest = sha256.clone();
        let advertised = filename.to_string();
        let counter = Arc::clone(&requests);
        let last_range = Arc::new(Mutex::new(None));
        let ranges = Arc::clone(&last_range);
        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    return;
                };
                counter.fetch_add(1, Ordering::SeqCst);
                let body = body.clone();
                let digest = digest.clone();
                let advertised = advertised.clone();
                let ranges = Arc::clone(&ranges);
                tokio::spawn(async move {
                    let _ = serve(stream, body, digest, advertised, behavior, ranges).await;
                });
            }
        });

        Self {
            endpoint: format!("http://127.0.0.1:{port}"),
            content,
            sha256,
            filename: filename.to_string(),
            requests,
            last_range,
        }
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn content(&self) -> &[u8] {
        &self.content
    }

    /// Hex SHA-256 of the served content, which is what the tree listing
    /// declares as the LFS digest.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// The byte range of the most recent content request, if it carried one.
    /// A resumed transfer asks for a range starting at what is already on disk.
    pub fn last_range(&self) -> Option<(u64, Option<u64>)> {
        *self.last_range.lock().unwrap()
    }

    /// Connections accepted so far. Zero proves a code path stayed offline.
    pub fn request_count(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }
}

pub fn hex_digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

async fn serve(
    mut stream: tokio::net::TcpStream,
    body: Vec<u8>,
    digest: String,
    filename: String,
    behavior: Behavior,
    ranges: RangeLog,
) -> std::io::Result<()> {
    let request = read_request(&mut stream).await?;
    let (method, target) = request_line(&request);

    if target.contains("/tree/") {
        let listing = tree_listing(&digest, &filename, body.len(), behavior);
        return write_response(
            &mut stream,
            200,
            "OK",
            &[("Content-Type", "application/json")],
            listing.as_bytes(),
        )
        .await;
    }

    if behavior == Behavior::NotFound {
        return write_response(&mut stream, 404, "Not Found", &[], b"").await;
    }

    let length = body.len();
    let etag = format!("\"{digest}\"");
    let length_header = length.to_string();
    if method == "HEAD" {
        let headers = [
            ("Content-Length", length_header.as_str()),
            ("ETag", etag.as_str()),
            ("x-repo-commit", "0123456789abcdef0123456789abcdef01234567"),
            ("Accept-Ranges", "bytes"),
        ];
        return write_head(&mut stream, 200, "OK", &headers).await;
    }

    let requested = match parse_range(&request) {
        Some(_) if behavior == Behavior::IgnoreRange => None,
        other => other,
    };
    if let Some((start, end)) = requested {
        *ranges.lock().unwrap() = Some((start as u64, end.map(|value| value as u64)));
    }

    match requested {
        Some((start, end)) => {
            let end = end
                .unwrap_or(length.saturating_sub(1))
                .min(length.saturating_sub(1));
            let slice = &body[start.min(length)..=end.max(start).min(length.saturating_sub(1))];
            let content_range = format!("bytes {start}-{end}/{length}");
            let slice_length = slice.len().to_string();
            let headers = [
                ("Content-Length", slice_length.as_str()),
                ("Content-Range", content_range.as_str()),
                ("ETag", etag.as_str()),
                ("x-repo-commit", "0123456789abcdef0123456789abcdef01234567"),
            ];
            write_body(
                &mut stream,
                206,
                "Partial Content",
                &headers,
                slice,
                behavior,
            )
            .await
        }
        None => {
            let headers = [
                ("Content-Length", length_header.as_str()),
                ("ETag", etag.as_str()),
                ("x-repo-commit", "0123456789abcdef0123456789abcdef01234567"),
                ("Accept-Ranges", "bytes"),
            ];
            write_body(&mut stream, 200, "OK", &headers, &body, behavior).await
        }
    }
}

/// The subset of the tree listing the digest lookup reads: one file entry whose
/// `lfs.sha256` is the declared digest.
fn tree_listing(digest: &str, filename: &str, size: usize, behavior: Behavior) -> String {
    let lfs = match behavior {
        Behavior::NoDeclaredDigest => String::new(),
        Behavior::WrongDeclaredDigest => {
            let wrong = "0".repeat(64);
            format!(",\"lfs\":{{\"sha256\":\"{wrong}\",\"size\":{size},\"pointerSize\":134}}")
        }
        _ => format!(",\"lfs\":{{\"sha256\":\"{digest}\",\"size\":{size},\"pointerSize\":134}}"),
    };
    format!(
        "[{{\"type\":\"file\",\"oid\":\"{digest}\",\"path\":\"{filename}\",\"size\":{size}{lfs}}}]"
    )
}

async fn read_request(stream: &mut tokio::net::TcpStream) -> std::io::Result<String> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 1024];
    loop {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if buffer.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }
    Ok(String::from_utf8_lossy(&buffer).into_owned())
}

fn request_line(request: &str) -> (&str, &str) {
    let first = request.lines().next().unwrap_or_default();
    let mut parts = first.split_whitespace();
    (
        parts.next().unwrap_or_default(),
        parts.next().unwrap_or_default(),
    )
}

/// Parse `Range: bytes=<start>-<end>`, where the end is optional.
fn parse_range(request: &str) -> Option<(usize, Option<usize>)> {
    let line = request
        .lines()
        .find(|line| line.to_ascii_lowercase().starts_with("range:"))?;
    let spec = line.split_once('=')?.1.trim();
    let (start, end) = spec.split_once('-')?;
    let start = start.trim().parse().ok()?;
    let end = end.trim();
    Some((
        start,
        if end.is_empty() {
            None
        } else {
            end.parse().ok()
        },
    ))
}

async fn write_head(
    stream: &mut tokio::net::TcpStream,
    code: u16,
    reason: &str,
    headers: &[(&str, &str)],
) -> std::io::Result<()> {
    stream
        .write_all(header_block(code, reason, headers).as_bytes())
        .await?;
    stream.flush().await
}

async fn write_response(
    stream: &mut tokio::net::TcpStream,
    code: u16,
    reason: &str,
    headers: &[(&str, &str)],
    body: &[u8],
) -> std::io::Result<()> {
    let length = body.len().to_string();
    let mut all: Vec<(&str, &str)> = headers.to_vec();
    all.push(("Content-Length", length.as_str()));
    stream
        .write_all(header_block(code, reason, &all).as_bytes())
        .await?;
    stream.write_all(body).await?;
    stream.flush().await
}

async fn write_body(
    stream: &mut tokio::net::TcpStream,
    code: u16,
    reason: &str,
    headers: &[(&str, &str)],
    body: &[u8],
    behavior: Behavior,
) -> std::io::Result<()> {
    stream
        .write_all(header_block(code, reason, headers).as_bytes())
        .await?;
    let body = match behavior {
        Behavior::TruncateAfter(n) => &body[..n.min(body.len())],
        _ => body,
    };
    stream.write_all(body).await?;
    stream.flush().await
}

fn header_block(code: u16, reason: &str, headers: &[(&str, &str)]) -> String {
    let mut block = format!("HTTP/1.1 {code} {reason}\r\nConnection: close\r\n");
    for (name, value) in headers {
        block.push_str(&format!("{name}: {value}\r\n"));
    }
    block.push_str("\r\n");
    block
}
