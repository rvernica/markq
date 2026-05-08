//! `markq` CLI. wires every v1 subcommand name as a `todo!()` stub
//! (so `markq --help` already prints the final surface) and implements
//! `markq inspect` against the LanceDB backend.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use markq_core::{default_dataset_path, Index};
use markq_index_lance::LanceIndex;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[derive(Parser, Debug)]
#[command(
    name = "markq",
    version,
    about = "Local-first markdown retrieval (BM25 + vector + RRF + rerank)"
)]
struct Cli {
    /// Path to the chunk dataset. Defaults to `~/.markq/chunks.lance`.
    #[arg(long, global = true)]
    dataset: Option<PathBuf>,

    /// Refuse network calls (model downloads, etc.). Currently parsed but
    /// not yet enforced — wires up once `hf-hub` model downloads land.
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

    /// Print chunks for a single markdown file (dev-only demo).
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

        // Every other v1 subcommand has its name + arg shape registered now
        // so `markq --help` matches the final surface; bodies land in their
        // respective milestones.
        Command::Index(_)
        | Command::Embed(_)
        | Command::Collection(_)
        | Command::Context(_)
        | Command::Search(_)
        | Command::Vsearch(_)
        | Command::Query(_)
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
        .with(fmt::layer().with_target(true))
        .init();
}
