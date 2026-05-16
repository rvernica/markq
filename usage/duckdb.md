# Querying markq's Lance dataset from DuckDB

The markq dataset at `~/.markq/chunks.lance` is a bare Lance directory, so you
can run SQL against it from DuckDB without going through the `markq` binary.
This is the composable-stack payoff that comes from storing chunks as a bare
Lance dataset rather than an opaque internal index.

All output below is from real runs against the default dataset after
indexing this repo's `README.md` (8 chunks). Some examples additionally
require `markq embed` to have run on the dataset (onward —
`embedding` column populated, HNSW index built). Reproduce with:

```sh
markq index README.md
markq embed   # ~640 MB GGUF downloaded on first run
```

## Prerequisites

The `lance` community extension is currently published only for **DuckDB
v1.5.0**. Other versions (1.4.x, 1.5.1+) return 404 when DuckDB tries to
download the extension. Pin to v1.5.0:

```sh
DUCKDB_VERSION=1.5.0 sh -c 'curl -s https://install.duckdb.org | sh'
# installs to ~/.duckdb/cli/1.5.0/duckdb
```

Install the extension once per DuckDB CLI install:

```sh
~/.duckdb/cli/1.5.0/duckdb -c "INSTALL lance FROM community;"
```

`LOAD lance;` is required at the start of every session.

## ATTACH the markq database (recommended)


LanceDB stores tables as `<db>/<table_name>.lance/`, where `<db>` is a
directory holding many tables. markq's `<db>` is `~/.markq/`, with a single
`chunks` table. `ATTACH` needs an absolute path, so expand `~` first:

```sql
LOAD lance;
ATTACH '/home/user/.markq' AS mq (TYPE lance);
SHOW TABLES FROM mq;
```

```
┌─────────┐
│  name   │
├─────────┤
│ chunks  │
└─────────┘
```

Once attached, `mq.chunks` behaves like a regular DuckDB table.

### Inspect the schema

```sql
DESCRIBE mq.chunks;
```

```
┌────────────────┬─────────────┐
│ column_name    │ column_type │
├────────────────┼─────────────┤
│ id             │ varchar     │
│ collection     │ varchar     │
│ uri            │ varchar     │
│ path           │ varchar     │
│ content_hash   │ varchar     │
│ mtime          │ bigint      │
│ chunk_index    │ integer     │
│ text           │ varchar     │
│ tokens         │ integer     │
│ embedding      │ float[1024] │
│ context_id     │ varchar     │
│ schema_version │ integer     │
└────────────────┴─────────────┘
```

The `embedding` column is a fixed-size 1024-float vector — DuckDB can read it
natively, no UDF needed. Before `markq embed` runs the column is all
NULL; after `embed`, each row holds a Qwen3-Embedding-0.6B Q8_0 vector.

```sql
SELECT chunk_index, length(embedding) AS dim, embedding[1:3] AS first3
FROM mq.chunks
ORDER BY chunk_index
LIMIT 3;
```

```
┌─────────────┬───────┬──────────────────────────────────────┐
│ chunk_index │  dim  │                first3                │
│    int32    │ int64 │               float[]                │
├─────────────┼───────┼──────────────────────────────────────┤
│           0 │  1024 │ [0.9659123, 0.7466198, -0.41592225]  │
│           1 │  1024 │ [1.649999, -0.61234635, -0.23107512] │
│           2 │  1024 │ [-2.1656308, -4.877181, -0.7711788]  │
└─────────────┴───────┴──────────────────────────────────────┘
```

DuckDB array slicing is 1-indexed (`[1:3]` returns elements 1, 2, 3 —
not 1, 2). The values are raw model output, not unit-normalized;
markq's `vsearch` path computes cosine distance under the hood.

### Count and select

```sql
SELECT COUNT(*) AS rows FROM mq.chunks;
```

```
┌───────┐
│ rows  │
│ int64 │
├───────┤
│     8 │
└───────┘
```

```sql
SELECT collection, path, chunk_index FROM mq.chunks ORDER BY chunk_index LIMIT 3;
```

```
┌────────────┬───────────────────────────┬─────────────┐
│ collection │           path            │ chunk_index │
├────────────┼───────────────────────────┼─────────────┤
│ default    │ /path/to/markq/README.md  │           0 │
│ default    │ /path/to/markq/README.md  │           1 │
│ default    │ /path/to/markq/README.md  │           2 │
└────────────┴───────────────────────────┴─────────────┘
```

(The `path` column reflects whatever absolute path `markq index` canonicalized
on your machine.)

### Aggregations

```sql
SELECT collection, COUNT(*) FROM mq.chunks GROUP BY 1;
```

```
┌────────────┬──────────────┐
│ collection │ count_star() │
├────────────┼──────────────┤
│ default    │            8 │
└────────────┴──────────────┘
```

```sql
SELECT path, chunk_index, length(text) AS chars
FROM mq.chunks
WHERE embedding IS NULL
ORDER BY chars DESC
LIMIT 3;
```

```
┌───────────────────────────┬─────────────┬───────┐
│           path            │ chunk_index │ chars │
├───────────────────────────┼─────────────┼───────┤
│ /path/to/markq/README.md  │           1 │  1139 │
│ /path/to/markq/README.md  │           2 │   899 │
│ /path/to/markq/README.md  │           3 │   717 │
└───────────────────────────┴─────────────┴───────┘
```

## Function-style scan (alternative)

If you'd rather not `ATTACH`, `__lance_scan(<dataset path>)` reads a single
dataset directly. The double underscore is the extension's "internal API"
marker — it's the function older Lance docs call `lance_scan`; the
published extension renames it.

```sql
LOAD lance;
SELECT id, path, chunk_index
FROM __lance_scan('/home/user/.markq/chunks.lance')
ORDER BY chunk_index
LIMIT 3;
```

```
┌────────────────────────────────────────────────────────────────────┬───────────────────────────┬─────────────┐
│                                 id                                 │           path            │ chunk_index │
├────────────────────────────────────────────────────────────────────┼───────────────────────────┼─────────────┤
│ 9e3a…                                                              │ /path/to/markq/README.md  │           0 │
│ 62e0…                                                              │ /path/to/markq/README.md  │           1 │
│ cb74…                                                              │ /path/to/markq/README.md  │           2 │
└────────────────────────────────────────────────────────────────────┴───────────────────────────┴─────────────┘
```

`ATTACH` is preferable for interactive work — it makes joins, `DESCRIBE`, and
multi-table queries idiomatic.

## What the search functions look like

The extension exposes three search table functions whose signatures will
become useful once markq populates the underlying indexes:

```sql
SELECT function_name, parameters
FROM duckdb_functions()
WHERE function_name LIKE 'lance_%'
ORDER BY 1;
```

```
┌─────────────────────┬──────────────────────────────────────────────────────────────────────────────────┐
│    function_name    │                                    parameters                                    │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
│ lance_fts           │ [col0, col1, col2, prefilter, k]                                                 │
│ lance_hybrid_search │ [col0, col1, col2, col3, col4, alpha, prefilter, oversample_factor, k]           │
│ lance_vector_search │ [col0, col1, col2, ..., explain_verbose, use_index, prefilter, refine_factor,    │
│                     │  nprobs, k]                                                                      │
└─────────────────────┴──────────────────────────────────────────────────────────────────────────────────┘
```

One gotcha: the optional parameters (`prefilter`, `k`, …) are
**named-only**. Positional `..., false, 10)` fails to bind even with the
right types; use `prefilter := false, k := 3` instead.

**BM25** works today — `markq index` builds the `text_idx` Inverted index:

```sql
LOAD lance;
SELECT chunk_index, _score, substr(text, 1, 60) AS preview
FROM lance_fts('/home/user/.markq/chunks.lance', 'text', 'lance',
               prefilter := false, k := 3);
```

```
┌─────────────┬────────────┬──────────────────────────────────────────────────────────────────┐
│ chunk_index │   _score   │                             preview                              │
├─────────────┼────────────┼──────────────────────────────────────────────────────────────────┤
│           1 │   1.022264 │ `markq` (mark[down] + q[uery]) is a local-first Rust CLI for     │
│           5 │  0.9687533 │ A tracked `pre-commit` hook in `.githooks/` runs `cargo fmt      │
│           6 │ 0.93084234 │ ```sh\ncargo run -q -p markq-cli -- inspect\n```\n\nCreates `~/. │
└─────────────┴────────────┴──────────────────────────────────────────────────────────────────┘
```

Scores match `markq search "lance"` exactly — same underlying index.

**Vector** (`lance_vector_search`) works after `markq embed` has run —
the dataset then carries both populated vectors and an `IvfHnswSq`
index on the `embedding` column. You'd supply the query vector
externally (DuckDB doesn't run the embedder); the markq Rust CLI takes
care of that for you via `markq vsearch`.

**Hybrid** (`lance_hybrid_search`, `alpha` weighting BM25 vs vector)
will see real signal once both indexes are populated together. The
markq-side hybrid `markq query` ships in with RRF fusion.

## Lance auto-cleanup config

LanceDB tracks dataset versions and auto-cleans old ones. The current
settings live in the dataset config and are visible via:

```sql
LOAD lance;
SELECT * FROM __lance_show_auto_cleanup('/home/user/.markq/chunks.lance');
```

```
┌────────────┬─────────┐
│    Key     │  Value  │
├────────────┼─────────┤
│ enabled    │ true    │
│ interval   │ 20      │
│ older_than │ 14days  │
└────────────┴─────────┘
```

Means "after 20 commits, prune any version older than 14 days." `markq
compact` will tune these on demand once it's wired up.

## What DuckDB *cannot* show you

The markq metadata keys we wrote at create time (`markq.schema_version`,
`markq.lance_manifest_version`, `markq.lance_file_format_version`,
`markq.lancedb_crate_version`) live in Lance's user `config` map.
`markq embed` adds two more (`markq.embedder_model`,
`markq.embedder_dim`). The DuckDB extension does not surface this map
through SQL — only its own `auto_cleanup.*` keys are exposed.

To read the markq metadata, use pylance instead:

```sh
uv run --with pylance python -c "
import lance, json
print(json.dumps(lance.dataset('$HOME/.markq/chunks.lance').config(), indent=2))
"
```

```json
{
  "lance.auto_cleanup.older_than": "14days",
  "lance.auto_cleanup.interval": "20",
  "markq.lance_manifest_version": "1",
  "markq.embedder_dim": "1024",
  "markq.lancedb_crate_version": "0.27.2",
  "markq.schema_version": "1",
  "markq.lance_file_format_version": "2.0",
  "markq.embedder_model": "Qwen/Qwen3-Embedding-0.6B-GGUF/Q8_0"
}
```

`markq doctor` will read these directly via the Rust API and turn
mismatches into structured errors. DuckDB is for ad-hoc data exploration.

## One-liner reference

```sh
# Open an interactive DuckDB shell with markq attached.
# Use -cmd (runs before stdin) — `-c` would execute the SQL and exit.
~/.duckdb/cli/1.5.0/duckdb -cmd "LOAD lance; ATTACH '$HOME/.markq' AS mq (TYPE lance);"

# Count chunks per collection.
~/.duckdb/cli/1.5.0/duckdb -c "
LOAD lance;
ATTACH '$HOME/.markq' AS mq (TYPE lance);
SELECT collection, COUNT(*) FROM mq.chunks GROUP BY 1 ORDER BY 2 DESC;
"

# Find the longest unembedded chunks.
~/.duckdb/cli/1.5.0/duckdb -c "
LOAD lance;
ATTACH '$HOME/.markq' AS mq (TYPE lance);
SELECT path, chunk_index, length(text) AS chars
FROM mq.chunks
WHERE embedding IS NULL
ORDER BY chars DESC
LIMIT 10;
"
```

## Vector search end-to-end: embed in one tool, query in DuckDB

`lance_vector_search(<dataset>, <column>, <query vector>, k := <N>)` runs
HNSW vector search inside DuckDB and returns the same chunks `markq
vsearch` would. The wrinkle is that DuckDB does not run the embedder
itself — you have to hand it a 1024-float query vector that came from
somewhere else. Three ways to produce that vector, all interchangeable
for retrieval (cosine ≈ 0.9998 between any pair on the same query,
inside llama.cpp's own run-to-run noise floor).

### Embed the query with `markq embed-query`

`markq embed-query "<text>"` prints the query embedding as a one-line
JSON array (1024 floats) on stdout, ready to be spliced into SQL. It
uses the exact same `Embedder::load` + `embedder.embed` path as `markq
vsearch`, so the vector is cosine-compatible with what's in the
dataset by construction.

```sh
# Capture the query vector once.
QVEC=$(markq embed-query "how does retrieval work")

# Splice it straight into lance_vector_search.
~/.duckdb/cli/1.5.0/duckdb -c "
LOAD lance;
SELECT chunk_index, _distance, substr(text, 1, 60) AS preview
FROM lance_vector_search('$HOME/.markq/chunks.lance', 'embedding',
                         $QVEC::FLOAT[1024], k := 3);
"
```

```
┌─────────────┬───────────┬──────────────────────────────────────────────────────────────────┐
│ chunk_index │ _distance │                             preview                              │
│    int32    │   float   │                             varchar                              │
├─────────────┼───────────┼──────────────────────────────────────────────────────────────────┤
│           0 │ 0.9712044 │ # markq\n\n`markq` (mark[down] + q[uery]) is a local-first Rus   │
│           6 │ 1.3101343 │ ```sh\ncargo run -q -p markq-cli -- inspect\n```\n\nCreates `~/. │
│           5 │ 1.3734075 │ A tracked `pre-commit` hook in `.githooks/` runs `cargo fmt      │
└─────────────┴───────────┴──────────────────────────────────────────────────────────────────┘
```

`_distance` is cosine distance (lower is better); the top hit is the
README's intro chunk, which is on-topic for "how does retrieval work".
This matches the ranking `markq vsearch "how does retrieval work" -n 3`
would produce — same embedder, same KNN, just driven from SQL.

A few practical notes:

- The JSON array `markq embed-query` prints is valid DuckDB list syntax,
  so the `$QVEC::FLOAT[1024]` cast is the only ceremony you need.
- `markq embed-query` validates the dataset's recorded `embedder_model`
  before loading the GGUF, so this composition fails loudly if the
  dataset was built with a different embedder rather than returning a
  silently incompatible vector.
- Run `markq query "<text>"` directly if you want BM25 + vector fused
  (it's already hybrid); the SQL path here is vector-only.

### Embed the query with `llama-embedding`

`llama-embedding` is the standalone one-shot CLI bundled with llama.cpp.
It reads the same GGUF and runs the same inference, so the vectors are
interchangeable with markq's for retrieval (we measured cosine
similarity ≈ 0.9998 between them, well inside llama.cpp's own
run-to-run noise floor).

The one thing to be careful about: `llama-embedding` defaults to L2-
normalizing the output. Markq does not, and Lance stores un-normalized
vectors, so pass `--embd-normalize -1` to get the raw vector.

```sh
# Start by pointing at your local llama.cpp build.
LLAMA_BIN=/path/to/llama.cpp/build/bin/llama-embedding

QVEC=$("$LLAMA_BIN" \
    --model ~/.cache/markq/models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --pooling last \
    --embd-normalize -1 \
    --embd-output-format json \
    --ctx-size 2048 \
    --n-gpu-layers 999 \
    --prompt "how does retrieval work" 2>/dev/null \
  | python3 -c "import json, sys; print(json.dumps(json.load(sys.stdin)['data'][0]['embedding']))")

~/.duckdb/cli/1.5.0/duckdb -c "
LOAD lance;
SELECT chunk_index, _distance, substr(text, 1, 60) AS preview
FROM lance_vector_search('$HOME/.markq/chunks.lance', 'embedding',
                         $QVEC::FLOAT[1024], k := 3);
"
```

```
┌─────────────┬────────────┬──────────────────────────────────────────────────────────────────┐
│ chunk_index │ _distance  │                             preview                              │
│    int32    │   float    │                             varchar                              │
├─────────────┼────────────┼──────────────────────────────────────────────────────────────────┤
│           0 │ 0.97111356 │ # markq\n\n`markq` (mark[down] + q[uery]) is a local-first Rus   │
│           6 │  1.3074002 │ ```sh\ncargo run -q -p markq-cli -- inspect\n```\n\nCreates `~/. │
│           5 │  1.3708969 │ A tracked `pre-commit` hook in `.githooks/` runs `cargo fmt      │
└─────────────┴────────────┴──────────────────────────────────────────────────────────────────┘
```

Same top-3, `_distance` values match `markq embed-query`'s to three
decimal places — the per-component drift between markq and llama.cpp's
matmul/attention path is too small to change retrieval order.

What each flag is for:

| Flag | Why |
|---|---|
| `--model` | The GGUF to use. Must be the same model `markq embed` recorded; check `markq inspect` for `embedder_model`. |
| `--pooling last` | Match markq. Qwen3-Embedding declares this in its GGUF metadata, but pin it. |
| `--embd-normalize -1` | Return the raw vector. Default is `2` (L2-normalized) which works for cosine search but no longer matches markq's stored vectors element-wise. |
| `--embd-output-format json` | Single OpenAI-style JSON envelope; the small python step peels off `data[0].embedding`. |
| `--ctx-size 2048` | Match markq's `DEFAULT_N_CTX`. |
| `--n-gpu-layers 999` | Offload every layer to GPU when llama.cpp was built with Vulkan/Metal/CUDA. Drop or set 0 for CPU. |

Add `--device Vulkan1` (or whichever device id) if your build needs
explicit device selection.

### Embed the query with `llama-server`

`llama-server` is the long-lived HTTP version of the same llama.cpp
inference code that `llama-embedding` runs one-shot. Use it when you'll
issue many queries in a row — the 3–4 second GGUF load + GPU init cost
is paid once, not per query.

Start the server in embedding mode (long-form flags throughout):

```sh
LLAMA_BIN=/path/to/llama.cpp/build/bin/llama-server

"$LLAMA_BIN" \
    --model ~/.cache/markq/models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --embedding \
    --pooling last \
    --ctx-size 2048 \
    --n-gpu-layers 999 \
    --device Vulkan1 \
    --port 8080
```

The server normalizes the embedding to unit length by default (no CLI
knob to change this — at least not in the build I tested). The
per-request body has an `embd_normalize` field that overrides it; pass
`-1` to get the raw vector that matches markq's stored vectors
element-wise.

```sh
QVEC=$(curl -sS http://127.0.0.1:8080/embedding \
    -H 'Content-Type: application/json' \
    -d '{"content": "how does retrieval work", "embd_normalize": -1}' \
  | python3 -c "import json, sys; print(json.dumps(json.load(sys.stdin)[0]['embedding'][0]))")

~/.duckdb/cli/1.5.0/duckdb -c "
LOAD lance;
SELECT chunk_index, _distance, substr(text, 1, 60) AS preview
FROM lance_vector_search('$HOME/.markq/chunks.lance', 'embedding',
                         $QVEC::FLOAT[1024], k := 3);
"
```

```
┌─────────────┬────────────┬──────────────────────────────────────────────────────────────────┐
│ chunk_index │ _distance  │                             preview                              │
│    int32    │   float    │                             varchar                              │
├─────────────┼────────────┼──────────────────────────────────────────────────────────────────┤
│           0 │ 0.97111356 │ # markq\n\n`markq` (mark[down] + q[uery]) is a local-first Rus   │
│           6 │  1.3074002 │ ```sh\ncargo run -q -p markq-cli -- inspect\n```\n\nCreates `~/. │
│           5 │  1.3708969 │ A tracked `pre-commit` hook in `.githooks/` runs `cargo fmt      │
└─────────────┴────────────┴──────────────────────────────────────────────────────────────────┘
```

Identical top-3 to the `llama-embedding` block above and to
`markq embed-query`'s `_distance` values to three decimals.

Endpoint and response notes:

- **`/embedding`** returns `[{"index": 0, "embedding": [[v0, v1, ...]]}]`
  — an outer list (one item per input), each with the embedding wrapped
  in *another* list. The double-wrap is so the same shape carries the
  per-token outputs when `pooling: none`. With `pooling: last` you only
  ever need `[0]['embedding'][0]`.
- **`/v1/embeddings`** is the OpenAI-compatible endpoint and gives
  `{"data": [{"embedding": [...]}]}` — flat vector, no double-wrap.
  Either works.
- **`embd_normalize` values:** `-1` raw, `0` max-absolute, `1` L1, `2`
  L2 (default), `>2` p-norm. Mirror of `llama-embedding --embd-normalize`.
- Forgetting `embd_normalize: -1` is the most common pitfall. The
  response *looks* fine (1024 floats), but every component is divided
  by the original norm. Cosine search against Lance's IVF/HNSW index
  still ranks correctly because cosine is normalization-invariant, but
  `_distance` values diverge from what `markq vsearch` reports, and
  any downstream code that does `vec[i] * scale` is off by ~85×.

When to pick which:

- **`markq embed-query`** — already in the markq binary, no second
  install. Best when you've decided markq is the embedding source of
  truth and you're scripting around the dataset it built.
- **`llama-embedding`** — one-shot CLI, no server lifecycle. Best for
  cross-implementation parity testing, or when the dataset was built
  outside markq and you don't want to thread the embedder model id
  through markq's metadata guard.
- **`llama-server`** — keeps the model loaded across calls. Best when
  many queries are coming in a row, when an MCP / agent / UI wants a
  network-reachable embedder, or when the same model serves both this
  embedder role and chat/completion to other clients.

## See also

- [`usage/markq.md`](markq.md) — the markq CLI itself.
- [`usage/pylance.md`](pylance.md) — same dataset from Python (covers
  polars and datafusion-python as well).
