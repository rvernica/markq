# Using the `markq` CLI

This file covers what the binary actually does today. The current build
ships the workspace skeleton + `markq inspect`; every other v1 subcommand
is registered (so `markq --help` shows the final surface) but stubs out at
runtime. Each subcommand will be lit up as the corresponding feature work
lands.

## Build

The workspace is at the repo root. There's no published binary yet:

```sh
cd /home/vernica/proj/markq
cargo build --release -p markq-cli
# or, for fast iteration:
cargo run -q -p markq-cli -- <args>
```

The binary is named `markq`:

```sh
./target/debug/markq --version
```

```
markq 0.1.0
```

## Global flags

```
--dataset <PATH>   Override ~/.markq/chunks.lance for this invocation.
--offline          Forward-compat parsing; not yet enforced.
RUST_LOG=...       Standard tracing-subscriber filter. Default is `warn`.
                   Examples: `markq=info`, `markq_index_lance=debug`.
```

## Subcommand surface

`markq --help` prints the final v1 surface. Bodies for everything except
`inspect` exit with `not implemented yet`:

```sh
markq --help
```

```
Local-first markdown retrieval (BM25 + vector + RRF + rerank)

Usage: markq [OPTIONS] <COMMAND>

Commands:
  index       Index a path into the default collection
  embed       Generate embeddings for unembedded rows
  collection  Manage collections
  context     Manage the context tree
  search      BM25 retrieval
  vsearch     Vector retrieval
  query       Hybrid retrieval (BM25 + vector + RRF)
  rerank      Hybrid retrieval + cross-encoder rerank
  get         Fetch one document by path or `#docid`
  multi-get   Fetch many documents by glob/csv/`#ids`
  compact     Reclaim space from tombstoned chunks
  doctor      Diagnose index, model, and dimension issues
  models      Manage the GGUF model cache
  watch       Filesystem watch + incremental reindex (`--features watch`)
  serve       Run the MCP server over stdio
  inspect     Print the dataset path, Arrow schema, row count, and recorded metadata
  status      Show index health, collection sizes, model state
  config      Show or edit the markq config
  help        Print this message or the help of the given subcommand(s)
```

Per-subcommand help works even on stubs — useful for previewing the final
flag surface:

```sh
markq query --help
```

```
Hybrid retrieval (BM25 + vector + RRF)

Usage: markq query [OPTIONS] <QUERY>

Arguments:
  <QUERY>

Options:
  -c, --collection <COLLECTION>
      --dataset <DATASET>        Path to the chunk dataset. Defaults to `~/.markq/chunks.lance`
      --json
      --offline                  Refuse network calls (model downloads, etc.). Currently parsed but not yet enforced — wires up once `hf-hub` model downloads land
      --files
      --all
  -n <TOP_K>                     [default: 10]
      --min-score <MIN_SCORE>
      --explain                  Per-stage timing + RRF contribution trace
```

Calling a stub fails fast with exit 1:

```sh
markq search "anything"
```

```
Error: not implemented yet
```

## What works today: `markq inspect`

`inspect` opens (or creates) the Lance dataset, prints the path, the Arrow
schema, the row count, and the four `markq.*` metadata keys recorded at
creation time:

```sh
markq inspect
```

```
dataset path:  /home/vernica/.markq/chunks.lance
arrow schema:
  id               Utf8                             nullable=false
  collection       Utf8                             nullable=false
  uri              Utf8                             nullable=false
  path             Utf8                             nullable=false
  content_hash     Utf8                             nullable=false
  mtime            Int64                            nullable=false
  chunk_index      Int32                            nullable=false
  text             Utf8                             nullable=false
  tokens           Int32                            nullable=true
  embedding        FixedSizeList(1024 x Float32)    nullable=true
  context_id       Utf8                             nullable=true
  schema_version   Int32                            nullable=false
rows:          0
schema_version:            1
lance_manifest_version:    1
lance_file_format_version: 2.0
lancedb_crate_version:     0.27.2
```

The default dataset path is `~/.markq/chunks.lance`. First run creates the
directory and the empty Lance dataset; subsequent runs open it in place.

### Use a throwaway dataset

```sh
markq --dataset /tmp/markq-demo.lance inspect
```

Same output structure, points at a different on-disk dataset. Useful for
local experiments without touching the default cache:

```sh
ls /tmp/markq-demo.lance
# _transactions/  _versions/
```

That bare directory is what tools like DuckDB and pylance read directly —
see [`usage/duckdb.md`](duckdb.md) and [`usage/pylance.md`](pylance.md).

### Tracing

Set `RUST_LOG` to see what the binary is doing under the hood:

```sh
RUST_LOG=markq=info markq --dataset /tmp/x.lance inspect
```

```
2026-05-05T20:31:21.159825Z  INFO markq_index_lance: creating chunks table path=/tmp/x.lance
dataset path:  /tmp/x.lance
...
```

`debug` level surfaces the open-vs-create branch and shows path resolution
logic. The default level is `warn` — set per-crate filters
(`markq_index_lance=debug`) for finer control.

## See also

- [`usage/duckdb.md`](duckdb.md) — read the dataset via SQL.
- [`usage/pylance.md`](pylance.md) — read the dataset from Python.
