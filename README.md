# markq

`markq` (mark[down] + q[uery]) is a local-first Rust CLI for indexing a
folder of markdown and answering queries against it with state-of-the-art
retrieval — BM25, vector cosine, RRF fusion, and a cross-encoder reranker —
**entirely on-device**. Single static binary, no external runtime.

## Status

Under active development; the core retrieval pipeline works end-to-end. The
current build ships:

- A Cargo workspace skeleton whose chunk schema already reserves the v1.5
  columns (`collection`, `uri`, `context_id`) and whose `Index` trait is
  backend-agnostic — so multi-collection routing and the context tree land
  as code, not as a Lance schema migration and re-embed.
- A working **LanceDB** backend that creates `~/.markq/chunks.lance` with
  the schema and writes versioning metadata (`markq.schema_version`,
  `markq.lance_manifest_version`, `markq.lance_file_format_version`,
  `markq.lancedb_crate_version`) into the dataset config at creation time.
- **Indexing** (`markq index`) — walk a markdown tree, chunk on heading
  boundaries with fenced-code protection, and write rows to Lance with a
  BM25 inverted index over the `text` column.
- **BM25 search** (`markq search`) with `--json` / `--files` / `-n` /
  `--min-score` output flags.
- **Embedding** (`markq embed`) using Qwen3-Embedding-0.6B Q8_0 on-device
  via `llama-cpp-2`, plus an HNSW vector index over the embeddings.
- **Vector search** (`markq vsearch`) — cosine KNN over the HNSW index.
- **Hybrid retrieval** (`markq query`) — concurrent BM25 + vector with
  weighted RRF fusion and an `--explain` per-stage timing + contribution
  trace.
- **Cross-encoder reranking** (`markq rerank`) — re-score first-stage
  candidates read on stdin (the `--json` output of `search` / `vsearch` /
  `query`) with the Qwen3-Reranker-0.6B Q8_0 cross-encoder and return them
  best-first; also available as an opt-in `--rerank` flag on `markq query`
  that reranks the fused results after fusion.
- **Model management** (`markq models pull` / `markq models ls`): fetch the
  embedder and reranker with resumable transfers, checked against the digest
  the repository declares, and list what is cached, how big it is, and whether
  anything vouches for it.
- **Diagnosis** (`markq doctor`): one read-only command reporting dataset
  presence, schema drift, embedder identity and dimension, index coverage as
  row counts, and each model's state, giving every finding one action that
  clears it. `--json` for scripts, `--verify` to hash every cached model.
- **Offline operation** (`--offline`, `MARKQ_OFFLINE`, `MARKQ_MODELS_DIR`):
  forbid network access outright, or point at a directory of pre-staged
  models. With every model cached, no command makes a request.
- **Query embedding** (`markq embed-query`) — print a query's embedding as
  a JSON array so external tools (DuckDB's `lance_vector_search`, pylance)
  can run their own vector search without loading a GGUF.

Index compaction (`compact`), document fetch (`get` / `multi-get`), the MCP
server, and the multi-collection / context-tree UX are pending. Their
subcommands are registered in the `clap` surface and exit with a structured
"not implemented yet" until their slice lands. See
[`usage/markq.md`](usage/markq.md) for the per-command status table.

## Layout

| Crate                  | What it does                                             |
|------------------------|----------------------------------------------------------|
| `markq-core`           | Chunk Arrow schema, `Index` trait, registry types, errors, dataset metadata keys, parameterized backend contract suite (`test-harness` feature) |
| `markq-index-lance`    | LanceDB-backed `Index` implementation                    |
| `markq-cli`            | The `markq` binary (clap, tracing)                       |
| `markq-chunker`        | Markdown chunker (heading splits, tree-sitter fenced-code protection, token budgeting) |
| `markq-inference`      | `llama-cpp-2` embedder + Qwen3-Reranker cross-encoder, each on a single owner thread + bounded channel; model cache, resumable fetch, digest sidecars |
| `markq-test-support`   | Test-only fixtures shared across crates (a local stand-in for the model hub) |

`usage/` contains runnable docs with real captured output:
[`markq.md`](usage/markq.md) (CLI surface), [`duckdb.md`](usage/duckdb.md) (SQL access), and
[`pylance.md`](usage/pylance.md) (Python access).

## Build

Stable Rust toolchain; building from source needs a C++ compiler and
CMake (for `llama-cpp-sys-2`'s vendored llama.cpp, which `markq-inference`
builds and links for the embedder).

```sh
cargo build --release -p markq-cli
cargo test --workspace
```

On Fedora 44, the Rust workspace inherits `BINDGEN_EXTRA_CLANG_ARGS` from
`.cargo/config.toml` so bindgen can find the clang resource directory at
`/usr/lib/clang/<major>/include`. Adjust the path if your clang version
differs.

### Git hooks

A tracked `pre-commit` hook in `.githooks/` runs `cargo fmt --all -- --check`
and blocks the commit if any Rust code isn't formatted. Enable it once per
clone:

```sh
git config core.hooksPath .githooks
```

## Quick demo

```sh
cargo run -q -p markq-cli -- inspect
```

Creates `~/.markq/chunks.lance` on first run and prints the dataset path,
Arrow schema, row count, and the recorded `markq.*` metadata. The dataset
is a bare Lance directory, readable from outside markq via DuckDB
([`usage/duckdb.md`](usage/duckdb.md)) or pylance
([`usage/pylance.md`](usage/pylance.md)).

## Related work

- [`tobi/qmd`](https://github.com/tobi/qmd) — TypeScript CLI in the same
  niche; markq draws inspiration from it for the overall pipeline shape
  (BM25 + vector + RRF + cross-encoder rerank). markq is **not** a port:
  defaults, behavior, storage layer, and surface diverge.

## License

[MIT](LICENSE).
