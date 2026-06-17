# markq

`markq` (mark[down] + q[uery]) is a local-first Rust CLI for indexing a
folder of markdown and answering queries against it with state-of-the-art
retrieval — BM25, vector cosine, RRF fusion, and a cross-encoder reranker —
**entirely on-device**. Single static binary, no external runtime.

## Status

Under active development; the core retrieval pipeline works end-to-end. The
current build ships:

- A Cargo workspace skeleton with the chunk schema and the `Index` trait
  pinned to their final v1.5 shape (so multi-collection routing and the
  context tree become additive changes later, not Lance migrations).
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
- **Query embedding** (`markq embed-query`) — print a query's embedding as
  a JSON array so external tools (DuckDB's `lance_vector_search`, pylance)
  can run their own vector search without loading a GGUF.

The cross-encoder reranker, model-management UX (`doctor` / `models`),
incremental reindex / `compact`, document fetch (`get` / `multi-get`), MCP
server, and the multi-collection / context-tree UX are pending — their
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
| `markq-inference`      | `llama-cpp-2` embedder on a single owner thread + bounded channel (reranker pending) |

`usage/` contains runnable docs with real captured output:
[`markq.md`](usage/markq.md) (CLI surface), [`duckdb.md`](usage/duckdb.md) (SQL access), and
[`pylance.md`](usage/pylance.md) (Python access).

## Build

Stable Rust toolchain; building from source needs a C++ compiler and
CMake (for `llama-cpp-sys-2`'s vendored llama.cpp — no-op for the current
build because `markq-inference` is a stub, but the dependency is in the
graph for forward compatibility).

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
on commits that touch `*.rs`. Enable it once per clone:

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
