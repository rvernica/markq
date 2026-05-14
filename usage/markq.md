# Using the `markq` CLI

This file covers what the binary actually does today. The current build
ships the workspace skeleton, `markq inspect`, `markq index` (markdown
walk + chunk + FTS index build), and `markq search` (BM25 over the FTS
index). Every other v1 subcommand is registered (so `markq --help` shows
the final surface) but stubs out at runtime. Each will be lit up as the
corresponding feature work lands.

All output below was captured against the default dataset at
`~/.markq/chunks.lance` after indexing this repo's `README.md`. Paths in
your own runs will reflect your local environment — two placeholders
appear throughout:

- `/home/user/.markq/chunks.lance` stands in for the expanded default
  dataset path; yours will show your actual `$HOME`.
- `/path/to/markq/README.md` stands in for the canonicalized absolute
  path of the indexed file; yours will reflect wherever you cloned the
  repo.

## Build

The workspace is at the repo root. There's no published binary yet:

```sh
cargo build --release --package markq-cli
# or, for fast iteration:
cargo run --quiet --package markq-cli -- <args>
```

The binary is named `markq`:

```sh
./target/release/markq --version
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

`markq --help` prints the final v1 surface. Bodies for unimplemented
commands exit with `not implemented yet`:

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
markq vsearch "anything"
```

```
Error: not implemented yet
```

## What works today

| Command | Status |
|---------|--------|
| `inspect` | ✅ Implemented |
| `index <path>` | ✅ Implemented (walk, chunk, FTS build; no embeddings yet) |
| `search <query>` | ✅ Implemented (BM25 via Lance inverted index) |
| `embed`, `vsearch`, `query`, `rerank` | Stub (`not implemented yet`) |
| `get`, `multi-get`, `compact`, `doctor` | Stub |
| `status`, `config` | Stub |
| `collection {add,list,remove}` | Stub |
| `context add`, `models {pull,ls}` | Stub |
| `watch`, `serve` | Stub |
| `search --explain`, `search --collection <name>` | Returns a structured "not implemented" error |

## `markq inspect`

`inspect` opens (or creates) the Lance dataset, prints the path, the
Arrow schema, the row count, and the four `markq.*` metadata keys
recorded at creation time:

```sh
markq inspect
```

```
dataset path:  /home/user/.markq/chunks.lance
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
rows:          8
schema_version:            1
lance_manifest_version:    1
lance_file_format_version: 2.0
lancedb_crate_version:     0.27.2
```

The default dataset path is `~/.markq/chunks.lance`. First run creates
the directory and the empty Lance dataset; subsequent runs open it in
place. The row count above (`8`) reflects the README walkthrough in the
next section; immediately after a fresh init it will read `rows: 0`.

## `markq index`

`index` canonicalizes the given path, walks it for `*.md` files, splits
each file into chunks, writes them into the Lance dataset, and (re)builds
the inverted index on the `text` column. The `embedding` column is left
NULL — vector embeddings land with `markq embed`.

Index this repo's `README.md` into the default dataset:

```sh
markq index README.md
```

```
indexed 1 file(s), 8 chunk(s) into /home/user/.markq/chunks.lance
```

`PATH` can also be a directory; markq walks it recursively for markdown.
A missing path fails fast:

```sh
markq index /tmp/does-not-exist
```

```
Error: canonicalize /tmp/does-not-exist

Caused by:
    No such file or directory (os error 2)
```

Passing `--collection <name>` is parsed but not yet enforced — everything currently
lands in the `default` collection. The chunks are addressable via the
`uri` column as `markq://<collection>/`.

## `markq search`

BM25 retrieval over the inverted index built by `index`. Tokenization
handles hyphenated terms via the underlying Lance tokenizer.

```sh
markq search "lance"
```

```
  1.   1.022  markq://default/#1
     `markq` (mark[down] + q[uery]) is a local-first Rust CLI for indexing a folder of markdown and answe…
  2.   0.969  markq://default/#5
     A tracked `pre-commit` hook in `.githooks/` runs `cargo fmt --all -- --check` on commits that touch …
  3.   0.931  markq://default/#6
     ```sh cargo run -q -p markq-cli -- inspect ``` Creates `~/.markq/chunks.lance` on first run and prin…
  4.   0.691  markq://default/#2
     Indexing, embedding, search, hybrid retrieval, the reranker, MCP, and the multi-collection / context…
```

Each line is `rank. score uri/#chunk_index` followed by a text preview.
Default `-n` is 10. Empty result sets print `(no results)`:

```sh
markq search "nonexistent_term_xyz"
```

```
(no results)
```

### Useful flags

- `-n <K>` — top-K (default 10).
- `--min-score <F>` — filter results below the BM25 score threshold.
- `--files` — print just the file paths (one per line, deduped).
- `--json` — emit the full result rows as a JSON array.
- `--all` — search across all collections (currently a no-op since only
  `default` exists).
- `--explain` and `--collection <name>` — recognized but stubbed; exit with
  a "not implemented yet" error rather than silently misbehaving.

```sh
markq search "lance" --files
```

```
/path/to/markq/README.md
```

```sh
markq search "lance" --json -n 1
```

```json
[
  {
    "chunk_index": 1,
    "id": "62e06bbf1b12bc85335b7c21b1ab62b4e42e7867f9fda8979ce0cc963bd1297a",
    "path": "/path/to/markq/README.md",
    "score": 1.022264003753662,
    "text": "`markq` (mark[down] + q[uery]) is a local-first Rust CLI for indexing a\nfolder of markdown ...",
    "uri": "markq://default/"
  }
]
```

## Use a throwaway dataset

`--dataset` overrides the default for a single invocation. Useful for
experiments without touching `~/.markq/chunks.lance`:

```sh
markq --dataset /tmp/markq-demo.lance inspect
markq --dataset /tmp/markq-demo.lance index README.md
markq --dataset /tmp/markq-demo.lance search "lance" -n 2
```

```
ls /tmp/markq-demo.lance
# _indices/  _transactions/  _versions/  data/
```

That bare directory is what tools like DuckDB and pylance read directly —
see [`usage/duckdb.md`](duckdb.md) and [`usage/pylance.md`](pylance.md).

## Tracing

Set `RUST_LOG` to see what the binary is doing under the hood. The
default level is `warn`, so an `info` filter on the markq crates is
usually silent on read paths; bump to `debug` to surface the
open-vs-create branch and path resolution:

```sh
RUST_LOG=markq_index_lance=debug markq inspect
```

```
2026-05-14T03:14:45.983053Z DEBUG markq_index_lance: opening existing chunks table path=/home/user/.markq/chunks.lance
dataset path:  /home/user/.markq/chunks.lance
...
```

A fresh init logs `creating chunks table` instead. Drop the per-crate
filter (`RUST_LOG=debug`) if you also want Lance's own
`dataset_events` info lines.

## See also

- [`usage/duckdb.md`](duckdb.md) — read the dataset via SQL.
- [`usage/pylance.md`](pylance.md) — read the dataset from Python.
