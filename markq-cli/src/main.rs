//! `markq` CLI. Every v1 subcommand name is wired up front (so `markq --help`
//! already prints the final surface); unimplemented ones bail at runtime while
//! the shipped commands run against the LanceDB backend.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use markq_core::{default_dataset_path, Index};
use markq_index_lance::LanceIndex;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use markq_cli::{embed_query, embedder_cmd, indexer, query, search, vsearch};

/// Version string with the compiled-in inference backend appended, so
/// `markq --version` reveals whether GPU offload is available without
/// having to `ldd` the binary or load a model.
#[cfg(feature = "vulkan")]
const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (backend: vulkan)");
#[cfg(all(feature = "cuda", not(feature = "vulkan")))]
const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (backend: cuda)");
#[cfg(not(any(feature = "vulkan", feature = "cuda")))]
const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (backend: cpu)");

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
    /// Hybrid retrieval + cross-encoder rerank.
    Rerank(QueryArgs),

    // === Document fetch ===
    /// Fetch one document by path or `#docid`.
    Get(GetArgs),
    /// Fetch many documents by glob/csv/`#ids`.
    MultiGet(MultiGetArgs),

    // === Index lifecycle ===
    /// Reclaim space from tombstoned chunks.
    Compact,
    /// Diagnose index, model, and dimension issues.
    Doctor,
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
    #[arg(short = 'n', default_value_t = 10)]
    top_k: usize,
    #[arg(long)]
    min_score: Option<f32>,
    /// Per-stage timing + RRF contribution trace.
    #[arg(long)]
    explain: bool,
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
    Pull { role: Option<String> },
    Ls,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    let dataset_path = cli.dataset.clone().unwrap_or_else(default_dataset_path);

    match cli.cmd {
        Command::Inspect => cmd_inspect(&dataset_path).await,
        Command::Chunk(args) => cmd_chunk(&args),
        Command::Index(args) => cmd_index(&dataset_path, &args).await,
        Command::Search(args) => cmd_search(&dataset_path, &args).await,
        Command::Embed(args) => cmd_embed(&dataset_path, &args).await,
        Command::EmbedQuery(args) => cmd_embed_query(&dataset_path, &args).await,
        Command::Vsearch(args) => cmd_vsearch(&dataset_path, &args).await,
        Command::Query(args) => cmd_query(&dataset_path, &args).await,

        // Every other v1 subcommand has its name + arg shape registered now
        // so `markq --help` matches the final surface; their bodies land
        // later.
        Command::Collection(_)
        | Command::Context(_)
        | Command::Rerank(_)
        | Command::Get(_)
        | Command::MultiGet(_)
        | Command::Compact
        | Command::Doctor
        | Command::Models(_)
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

    let outcome = query::run_query(&idx, &args.query, k, &opts, args.explain).await?;

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

async fn cmd_search(dataset_path: &std::path::Path, args: &QueryArgs) -> Result<()> {
    if args.collection.is_some() {
        anyhow::bail!("collection filtering is not implemented yet; omit -c for now");
    }
    if args.explain {
        anyhow::bail!("--explain is not implemented yet");
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
