//! `markq` CLI. Every v1 subcommand name is wired up front (so `markq --help`
//! already prints the final surface); unimplemented ones bail at runtime while
//! the shipped commands run against the LanceDB backend.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use markq_core::{default_dataset_path, Index};
use markq_index_lance::LanceIndex;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use markq_cli::{embed_query, embedder_cmd, indexer, models, query, rerank, search, vsearch};

/// Version string with the compiled-in inference backend appended, so
/// `markq --version` reveals whether GPU offload is available without
/// having to `ldd` the binary or load a model.
#[cfg(feature = "vulkan")]
const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (backend: vulkan)");
#[cfg(all(feature = "cuda", not(feature = "vulkan")))]
const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (backend: cuda)");
#[cfg(not(any(feature = "vulkan", feature = "cuda")))]
const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (backend: cpu)");

/// Render a byte count the way a person reads a model file's size.
fn human_size(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.1} GiB", bytes / GIB)
    } else {
        format!("{:.0} MiB", bytes / MIB)
    }
}

/// The environment variable that forbids network access.
const OFFLINE_VARIABLE: &str = "MARKQ_OFFLINE";

/// Which setting forbade network access, for an error that can name it. `None`
/// means the network may be used.
type OfflineSetting = Option<String>;

/// Resolve the offline setting: the flag beats the variable, which beats the
/// default of allowing the network.
fn resolve_offline(flag: bool, value: Option<&str>) -> markq_core::Result<OfflineSetting> {
    if flag {
        return Ok(Some("--offline".to_string()));
    }
    Ok(if parse_offline_value(value)? {
        Some(OFFLINE_VARIABLE.to_string())
    } else {
        None
    })
}

/// Interpret one value of the offline variable.
///
/// Takes the value rather than reading the environment, so callers and tests
/// share one definition of what is accepted without mutating process-global
/// state. An unrecognized value is an error: silently reading `no` as "offline
/// on" would forbid the network the user asked for.
fn parse_offline_value(value: Option<&str>) -> markq_core::Result<bool> {
    let Some(value) = value else {
        return Ok(false);
    };
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" => Ok(true),
        "0" | "false" => Ok(false),
        _ => Err(markq_core::Error::InvalidOfflineValue {
            variable: OFFLINE_VARIABLE,
            value: value.to_string(),
            accepted: "1, true, 0, false".to_string(),
        }),
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "markq",
    version = VERSION,
    about = "Local-first markdown retrieval (BM25 + vector + RRF + rerank)"
)]
struct Cli {
    /// Path to the chunk dataset. Defaults to `~/.markq/chunks.lance`.
    #[arg(long, global = true)]
    dataset: Option<PathBuf>,

    /// Forbid all network access. Also set by `MARKQ_OFFLINE=1`.
    #[arg(long, global = true)]
    offline: bool,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    // === Corpus management ===
    /// Index a path into the default collection.
    Index(IndexArgs),
    /// Generate embeddings for unembedded rows.
    Embed(EmbedArgs),
    /// Embed one query string and print the vector as JSON.
    EmbedQuery(EmbedQueryArgs),

    /// Manage collections.
    #[command(subcommand)]
    Collection(CollectionCmd),
    /// Manage the context tree.
    #[command(subcommand)]
    Context(ContextCmd),

    // === Retrieval ===
    /// BM25 retrieval.
    Search(QueryArgs),
    /// Vector retrieval.
    Vsearch(QueryArgs),
    /// Hybrid retrieval (BM25 + vector + RRF).
    Query(QueryArgs),
    /// Cross-encoder rerank of stdin candidates.
    Rerank(RerankArgs),

    // === Document fetch ===
    /// Fetch one document by path or `#docid`.
    Get(GetArgs),
    /// Fetch many documents by glob/csv/`#ids`.
    MultiGet(MultiGetArgs),

    // === Index lifecycle ===
    /// Reclaim space from tombstoned chunks.
    Compact,
    /// Diagnose index, model, and dimension issues.
    Doctor(DoctorArgs),
    /// Manage the GGUF model cache.
    #[command(subcommand)]
    Models(ModelsCmd),
    /// Filesystem watch + incremental reindex (`--features watch`).
    Watch(WatchArgs),

    /// Print chunks for a single markdown file (dev-only chunker demo).
    #[command(hide = true)]
    Chunk(ChunkArgs),

    // === Other ===
    /// Run the MCP server over stdio.
    Serve,
    /// Print the dataset path, Arrow schema, row count, and recorded metadata.
    Inspect,
    /// Show index health, collection sizes, model state.
    Status,
    /// Show or edit the markq config.
    Config,
}

#[derive(Args, Debug)]
struct IndexArgs {
    path: Option<PathBuf>,
    /// Optional collection name.
    #[arg(short = 'c', long)]
    collection: Option<String>,
}

#[derive(Args, Debug)]
struct EmbedArgs {
    #[arg(short = 'c', long)]
    collection: Option<String>,
}

#[derive(Args, Debug)]
struct EmbedQueryArgs {
    query: String,
}

#[derive(Args, Debug)]
struct QueryArgs {
    query: String,
    #[arg(short = 'c', long)]
    collection: Option<String>,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    files: bool,
    #[arg(long)]
    all: bool,
    #[arg(short = 'n', long = "top-k", default_value_t = 10)]
    top_k: usize,
    #[arg(long)]
    min_score: Option<f32>,
    /// Per-stage timing + RRF contribution trace.
    #[arg(long)]
    explain: bool,
    /// Cross-encoder rerank of the fused candidate pool (query only).
    /// Results are then ordered by cross-encoder relevance instead of the
    /// fusion score, and each hit's score becomes that relevance
    /// probability in [0, 1]; `--min-score` (if given) filters on this
    /// probability, not the fusion score. Only the top 64 fused candidates
    /// are reranked, so combining this with a larger `--top-k` (`-n`) or
    /// `--all` is capped at 64 results.
    #[arg(long)]
    rerank: bool,
}

#[derive(Args, Debug)]
struct RerankArgs {
    /// The query to score every stdin candidate against.
    #[arg(long)]
    query: String,
    /// Keep only the top `k` candidates after reordering.
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long)]
    json: bool,
    /// Override the default retrieval instruction sent to the reranker.
    #[arg(long)]
    instruction: Option<String>,
}

#[derive(Args, Debug)]
struct GetArgs {
    target: String,
    #[arg(long)]
    full: bool,
}

#[derive(Args, Debug)]
struct MultiGetArgs {
    targets: Vec<String>,
}

#[derive(Args, Debug)]
struct ChunkArgs {
    /// Markdown file to chunk.
    file: PathBuf,
    /// Print chunk text bodies (default: only headers + token counts).
    #[arg(long)]
    text: bool,
    /// Override the default 900-token target.
    #[arg(long)]
    max_tokens: Option<usize>,
    /// Override the default 135-token (~15%) overlap.
    #[arg(long)]
    overlap_tokens: Option<usize>,
}

#[derive(Args, Debug)]
struct WatchArgs {
    collection: Option<String>,
}

#[derive(Args, Debug)]
struct DoctorArgs {
    #[arg(long)]
    json: bool,
    /// Also compute each cached model's full digest, catching a same-length
    /// alteration the default size-only check can't see. Slower than the
    /// default run.
    #[arg(long)]
    verify: bool,
}

#[derive(Subcommand, Debug)]
enum CollectionCmd {
    Add {
        path: PathBuf,
        #[arg(long)]
        name: String,
    },
    List,
    Remove {
        name: String,
    },
}

#[derive(Subcommand, Debug)]
enum ContextCmd {
    Add { uri: String, description: String },
}

#[derive(Subcommand, Debug)]
enum ModelsCmd {
    Pull {
        role: Option<String>,
    },
    Ls {
        #[arg(long)]
        json: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    let dataset_path = cli.dataset.clone().unwrap_or_else(default_dataset_path);
    let offline = resolve_offline(cli.offline, std::env::var(OFFLINE_VARIABLE).ok().as_deref())?;
    // Model loading deep in the embedder and reranker paths reads the variable
    // rather than taking a parameter, so the flag is published here, once, at
    // startup. Without this the flag would be global in name only.
    if cli.offline {
        std::env::set_var(OFFLINE_VARIABLE, "1");
    }

    match cli.cmd {
        Command::Inspect => cmd_inspect(&dataset_path).await,
        Command::Chunk(args) => cmd_chunk(&args),
        Command::Index(args) => cmd_index(&dataset_path, &args).await,
        Command::Search(args) => cmd_search(&dataset_path, &args).await,
        Command::Embed(args) => cmd_embed(&dataset_path, &args).await,
        Command::EmbedQuery(args) => cmd_embed_query(&dataset_path, &args).await,
        Command::Vsearch(args) => cmd_vsearch(&dataset_path, &args).await,
        Command::Query(args) => cmd_query(&dataset_path, &args).await,
        Command::Rerank(args) => cmd_rerank(&args).await,
        Command::Models(cmd) => cmd_models(cmd, offline.as_deref()).await,
        Command::Doctor(args) => cmd_doctor(&dataset_path, &args).await,

        // Every other v1 subcommand has its name + arg shape registered now
        // so `markq --help` matches the final surface; their bodies land
        // later.
        Command::Collection(_)
        | Command::Context(_)
        | Command::Get(_)
        | Command::MultiGet(_)
        | Command::Compact
        | Command::Watch(_)
        | Command::Serve
        | Command::Status
        | Command::Config => {
            anyhow::bail!("not implemented yet");
        }
    }
}

async fn cmd_index(dataset_path: &std::path::Path, args: &IndexArgs) -> Result<()> {
    if args.collection.is_some() {
        anyhow::bail!("multi-collection indexing is not implemented yet; omit -c for now");
    }
    let root = args.path.clone().unwrap_or_else(|| PathBuf::from("."));
    let idx = LanceIndex::open_or_create(dataset_path)
        .await
        .context("open or create dataset")?;
    let report = indexer::run_index(&idx, &root).await?;
    println!(
        "indexed {} file(s), {} chunk(s) ({} skipped, {} removed) into {}",
        report.files,
        report.chunks,
        report.skipped,
        report.removed,
        idx.path().display()
    );
    Ok(())
}

async fn cmd_embed(dataset_path: &std::path::Path, args: &EmbedArgs) -> Result<()> {
    if args.collection.is_some() {
        anyhow::bail!("multi-collection embed is not implemented yet; omit -c for now");
    }
    let idx = LanceIndex::open_or_create(dataset_path)
        .await
        .context("open or create dataset")?;
    let report = embedder_cmd::run_embed(&idx).await?;
    println!(
        "embedded {} row(s) over {} batch(es) (model={}, dim={})",
        report.rows, report.batches, report.model_id, report.dim,
    );
    Ok(())
}

async fn cmd_embed_query(dataset_path: &std::path::Path, args: &EmbedQueryArgs) -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    embed_query::run_embed_query(dataset_path, &args.query, &mut out).await
}

async fn cmd_vsearch(dataset_path: &std::path::Path, args: &QueryArgs) -> Result<()> {
    if args.collection.is_some() {
        anyhow::bail!("collection filtering is not implemented yet; omit -c for now");
    }
    if args.explain {
        anyhow::bail!("--explain is not implemented yet");
    }
    if args.rerank {
        anyhow::bail!("--rerank is only supported on `markq query`");
    }
    let format = match (args.json, args.files) {
        (true, true) => anyhow::bail!("--json and --files are mutually exclusive"),
        (true, false) => search::Format::Json,
        (false, true) => search::Format::Files,
        (false, false) => search::Format::Table,
    };
    let opts = search::SearchOptions {
        top_k: if args.all { None } else { Some(args.top_k) },
        min_score: args.min_score,
    };
    let idx = LanceIndex::open(dataset_path)
        .await
        .context("open dataset")?;
    let k = match opts.top_k {
        Some(k) => k,
        None => idx
            .count_rows()
            .await
            .context("count rows for --all")?
            .max(1),
    };
    let hits = vsearch::run_vsearch(&idx, &args.query, k, &opts).await?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    search::write_results(&mut out, &hits, format)?;
    Ok(())
}

async fn cmd_query(dataset_path: &std::path::Path, args: &QueryArgs) -> Result<()> {
    if args.collection.is_some() {
        anyhow::bail!("collection filtering is not implemented yet; omit -c for now");
    }
    let format = match (args.json, args.files) {
        (true, true) => anyhow::bail!("--json and --files are mutually exclusive"),
        (true, false) => search::Format::Json,
        (false, true) => search::Format::Files,
        (false, false) => search::Format::Table,
    };
    let opts = search::SearchOptions {
        top_k: if args.all { None } else { Some(args.top_k) },
        min_score: args.min_score,
    };

    let idx = LanceIndex::open(dataset_path)
        .await
        .context("open dataset")?;
    let k = match opts.top_k {
        Some(k) => k,
        None => idx
            .count_rows()
            .await
            .context("count rows for --all")?
            .max(1),
    };

    let outcome = query::run_query(&idx, &args.query, k, &opts, args.explain, args.rerank).await?;

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    search::write_results(&mut out, &outcome.hits, format)?;

    if let Some(trace) = &outcome.explain {
        let stderr = std::io::stderr();
        let mut err = stderr.lock();
        query::write_explain(&mut err, trace, &outcome.hits)?;
    }
    Ok(())
}

async fn cmd_rerank(args: &RerankArgs) -> Result<()> {
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    rerank::run_rerank(
        &mut input,
        &mut out,
        &args.query,
        args.top_k,
        args.instruction.as_deref(),
        args.json,
    )
    .await
}

async fn cmd_models(cmd: ModelsCmd, offline: Option<&str>) -> Result<()> {
    match cmd {
        ModelsCmd::Pull { role } => {
            let models_dir = markq_inference::models_dir()?;
            let client = markq_inference::HFClient::new().context("HFClient::new")?;
            let outcomes = models::run_pull(role.as_deref(), &models_dir, &client, offline).await?;

            let mut any_failed = false;
            for outcome in &outcomes {
                match &outcome.result {
                    Ok(entry) => println!(
                        "{}: ready ({:?}{})",
                        outcome.role.as_str(),
                        entry.state,
                        if outcome.transferred {
                            ", fetched"
                        } else {
                            ", already cached"
                        }
                    ),
                    Err(message) => {
                        any_failed = true;
                        println!("{}: failed: {message}", outcome.role.as_str());
                    }
                }
            }
            if any_failed {
                anyhow::bail!("not every requested model could be made ready");
            }
            Ok(())
        }
        ModelsCmd::Ls { json } => {
            let models_dir = markq_inference::models_dir()?;
            let rows = models::ls_rows(&models_dir)?;
            if json {
                serde_json::to_writer_pretty(std::io::stdout(), &rows)?;
                println!();
            } else {
                println!("{:<10} {:<11} {:>9}  MODEL", "ROLE", "STATE", "SIZE");
                for row in &rows {
                    println!(
                        "{:<10} {:<11} {:>9}  {}",
                        row.role,
                        row.state,
                        row.size_bytes.map(human_size).unwrap_or_else(|| "-".into()),
                        row.model_id,
                    );
                    println!("{:<34}{}", "", row.filename);
                }
            }
            Ok(())
        }
    }
}

async fn cmd_doctor(dataset_path: &std::path::Path, args: &DoctorArgs) -> Result<()> {
    let models_dir = markq_inference::models_dir()?;
    let findings = markq_cli::doctor::run_doctor(dataset_path, &models_dir, args.verify).await?;

    if args.json {
        serde_json::to_writer_pretty(std::io::stdout(), &findings)?;
        println!();
    } else {
        for f in &findings {
            let tag = format!("[{}]", f.severity);
            let line = format!("{tag:<10} {:<24} {}", f.check, f.next_step);
            println!("{}", line.trim_end());
            if let Some(observed) = &f.observed {
                println!("           observed: {observed}");
            }
            if let Some(expected) = &f.expected {
                println!("           expected: {expected}");
            }
        }
    }

    if markq_cli::doctor::any_error(&findings) {
        anyhow::bail!("doctor found one or more error-severity findings");
    }
    Ok(())
}

async fn cmd_search(dataset_path: &std::path::Path, args: &QueryArgs) -> Result<()> {
    if args.collection.is_some() {
        anyhow::bail!("collection filtering is not implemented yet; omit -c for now");
    }
    if args.explain {
        anyhow::bail!("--explain is not implemented yet");
    }
    if args.rerank {
        anyhow::bail!("--rerank is only supported on `markq query`");
    }

    let format = match (args.json, args.files) {
        (true, true) => anyhow::bail!("--json and --files are mutually exclusive"),
        (true, false) => search::Format::Json,
        (false, true) => search::Format::Files,
        (false, false) => search::Format::Table,
    };
    let opts = search::SearchOptions {
        top_k: if args.all { None } else { Some(args.top_k) },
        min_score: args.min_score,
    };

    let idx = LanceIndex::open(dataset_path)
        .await
        .context("open dataset")?;
    // For `--all`, size `k` from the row count so the BM25 query is not
    // silently truncated by a hardcoded budget. A query can't match more
    // chunks than exist.
    let k = match opts.top_k {
        Some(k) => k,
        None => idx
            .count_rows()
            .await
            .context("count rows for --all")?
            .max(1),
    };
    let raw = idx
        .bm25(&args.query, k, None)
        .await
        .context("bm25 search")?;
    let hits = search::apply_filters(raw, &opts);

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    search::write_results(&mut out, &hits, format)?;
    Ok(())
}

fn cmd_chunk(args: &ChunkArgs) -> Result<()> {
    use markq_chunker::{chunk_markdown, ApproxTokenizer, ChunkOptions};

    let src = std::fs::read_to_string(&args.file)
        .with_context(|| format!("read {}", args.file.display()))?;

    let mut opts = ChunkOptions::default();
    if let Some(m) = args.max_tokens {
        opts.max_tokens = m;
    }
    if let Some(o) = args.overlap_tokens {
        opts.overlap_tokens = o;
    }

    let chunks = chunk_markdown(&src, &opts, &ApproxTokenizer);
    println!("file:   {}", args.file.display());
    println!("bytes:  {}", src.len());
    println!(
        "chunks: {}  (max_tokens={}, overlap_tokens={})",
        chunks.len(),
        opts.max_tokens,
        opts.overlap_tokens
    );
    for c in &chunks {
        println!(
            "[{:>3}] bytes {:>7}..{:<7}  tokens={}",
            c.index, c.start_byte, c.end_byte, c.token_count
        );
        if args.text {
            println!("---");
            println!("{}", c.text);
            println!("---");
        }
    }
    Ok(())
}

async fn cmd_inspect(dataset_path: &std::path::Path) -> Result<()> {
    let idx = LanceIndex::open_or_create(dataset_path)
        .await
        .context("open or create dataset")?;

    println!("dataset path:  {}", idx.path().display());

    let schema = idx.arrow_schema();
    println!("arrow schema:");
    for f in schema.fields() {
        println!(
            "  {:<16} {:<32} nullable={}",
            f.name(),
            format!("{}", f.data_type()),
            f.is_nullable()
        );
    }

    let rows = idx.count_rows().await?;
    println!("rows:          {rows}");

    let md = idx.metadata().await?;
    println!("schema_version:            {}", md.schema_version);
    println!("lance_manifest_version:    {}", md.lance_manifest_version);
    println!(
        "lance_file_format_version: {}",
        md.lance_file_format_version
    );
    println!("lancedb_crate_version:     {}", md.lancedb_crate_version);
    if let Some(model) = &md.embedder_model {
        println!("embedder_model:            {model}");
    }
    if let Some(dim) = md.embedder_dim {
        println!("embedder_dim:              {dim}");
    }
    Ok(())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_target(true).with_writer(std::io::stderr))
        .init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_flag_accepted_on_every_subcommand() {
        // A user should never have to know which subcommands reach the network.
        for argv in [
            vec!["markq", "--offline", "inspect"],
            vec!["markq", "inspect", "--offline"],
            vec!["markq", "search", "needle", "--offline"],
            vec!["markq", "query", "needle", "--offline"],
            vec!["markq", "doctor", "--offline"],
            vec!["markq", "models", "ls", "--offline"],
        ] {
            let parsed = Cli::try_parse_from(&argv)
                .unwrap_or_else(|error| panic!("{argv:?} rejected: {error}"));
            assert!(parsed.offline, "{argv:?} did not set offline");
        }
    }

    #[test]
    fn flag_overrides_env_which_overrides_default() {
        assert_eq!(resolve_offline(false, None).unwrap(), None);
        assert_eq!(
            resolve_offline(false, Some("1")).unwrap().as_deref(),
            Some("MARKQ_OFFLINE")
        );
        assert_eq!(
            resolve_offline(true, None).unwrap().as_deref(),
            Some("--offline")
        );
        // The flag is the stronger statement, so it wins over a variable that
        // says otherwise.
        assert_eq!(
            resolve_offline(true, Some("0")).unwrap().as_deref(),
            Some("--offline")
        );
        assert_eq!(resolve_offline(false, Some("false")).unwrap(), None);
    }

    #[test]
    fn offline_env_accepts_documented_values_case_insensitively() {
        for on in ["1", "true", "TRUE", "True"] {
            assert!(
                parse_offline_value(Some(on)).unwrap(),
                "{on:?} should mean offline"
            );
        }
        for off in ["0", "false", "FALSE", "False"] {
            assert!(
                !parse_offline_value(Some(off)).unwrap(),
                "{off:?} should mean online"
            );
        }
        assert!(
            !parse_offline_value(None).unwrap(),
            "unset should mean online"
        );
    }

    #[test]
    fn offline_env_rejects_unrecognized_value() {
        let error = parse_offline_value(Some("maybe")).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("MARKQ_OFFLINE"), "got: {message}");
        assert!(message.contains("maybe"), "got: {message}");
        // The message has to say what would have worked.
        assert!(message.contains("true"), "got: {message}");
    }
}
