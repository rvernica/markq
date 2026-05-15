# Querying markq's Lance dataset from pylance (Python)

`pylance` is the Python binding for the Lance format. It opens the same
on-disk dataset markq writes — `~/.markq/chunks.lance` — and exposes the full
Lance API (schema, scan, SQL, time-travel, statistics, index management).

It's the right tool whenever you want **richer access than DuckDB SQL** —
specifically:
- Reading markq's own `markq.*` config keys (DuckDB doesn't surface the user
  config map; pylance does).
- Time-traveling to a prior dataset version.
- Inspecting fragments, indices, or storage statistics.
- Round-tripping to pandas / NumPy for ML work.

All output below was captured against the default dataset after indexing
this repo's `README.md`. Reproduce it with:

```sh
markq index README.md
markq embed   # optional; some examples below assume the embedding column is populated
```

`index` populates 8 chunks and builds the FTS inverted index on `text`.
`embed` fills the `embedding` column (1024-float vectors from
Qwen3-Embedding-0.6B Q8_0) and adds an HNSW vector index alongside.

## Prerequisites

```sh
uv run --with pylance --with pyarrow python
# or, for ad-hoc snippets:
uv run --with pylance --with pyarrow python -c '...'
```

Add `--with pandas`, `--with polars`, or `--with datafusion` for the
sections below that use those libraries; `uv` only installs what you
declare per invocation.

Pylance is currently `4.0.x`; markq pins lancedb-rust to `0.27.2`, which
embeds the same Lance file format (`2.0`).

## Open the dataset

```python
import os, lance
ds = lance.dataset(os.path.expanduser("~/.markq/chunks.lance"))
print("uri:           ", ds.uri)
print("version:       ", ds.version)
print("latest_version:", ds.latest_version)
print("count_rows:    ", ds.count_rows())
```

```
uri:            /home/user/.markq/chunks.lance
version:        4
latest_version: 4
count_rows:     8
```

`version` increments on every write commit. After a fresh init + one
`markq index README.md` you'll see four commits: two for `create_table` +
metadata `update_config`, then two more for the append + FTS index build.

## Read the markq metadata (the headline payoff over DuckDB)

```python
import os, lance, json
ds = lance.dataset(os.path.expanduser("~/.markq/chunks.lance"))
print(json.dumps(ds.config(), indent=2))
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

`markq.embedder_model` and `markq.embedder_dim` appear only after
`markq embed` has run; before that the dict has the other six entries.
This is what `markq doctor` will read via the Rust API and turn into
structured errors on mismatch. From Python you can sanity-check it without
going through the binary.

`config()` is a method (not a property) in pylance ≥ 4.0.

## Inspect the schema

```python
import os, lance
ds = lance.dataset(os.path.expanduser("~/.markq/chunks.lance"))
print(ds.schema)
```

```
id: string not null
collection: string not null
uri: string not null
path: string not null
content_hash: string not null
mtime: int64 not null
chunk_index: int32 not null
text: string not null
tokens: int32
embedding: fixed_size_list<item: float>[1024]
  child 0, item: float
context_id: string
schema_version: int32 not null
```

`ds.schema` is the Arrow schema. `ds.lance_schema` is the native Lance schema
(carries Lance-specific field metadata, useful when debugging encoding
choices). `ds.schema_metadata` is a dict of schema-level metadata — empty
under markq because we put everything in `config()` instead.

## Read rows

### As an Arrow Table

```python
import os, lance
ds = lance.dataset(os.path.expanduser("~/.markq/chunks.lance"))
tbl = ds.to_table()
print(type(tbl).__name__)
print(tbl.num_rows, "rows,", tbl.num_columns, "columns")
```

```
Table
8 rows, 12 columns
```

### As a pandas DataFrame

Run with `uv run --with pylance --with pyarrow --with pandas python`.

```python
import os, lance
ds = lance.dataset(os.path.expanduser("~/.markq/chunks.lance"))
df = ds.to_table().to_pandas()
print(df.shape)
print(df[["path", "chunk_index"]].head())
```

```
(8, 12)
                        path  chunk_index
0  /path/to/markq/README.md            0
1  /path/to/markq/README.md            1
2  /path/to/markq/README.md            2
3  /path/to/markq/README.md            3
4  /path/to/markq/README.md            4
```

```python
print(df.dtypes)
```

```
id                   str
collection           str
uri                  str
path                 str
content_hash         str
mtime              int64
chunk_index        int32
text                 str
tokens             int32
embedding         object   # NumPy array per row, shape (1024,) — currently None
context_id           str
schema_version     int32
```

The `embedding` column round-trips as Python objects (NumPy `float32` arrays
of length 1024); use `np.stack(df["embedding"])` for a 2D matrix:

```python
import numpy as np
mat = np.stack(df["embedding"])
print(mat.shape, mat.dtype)
# (8, 1024) float32
```

Each row is a raw model output, not unit-normalized. Cosine-similarity
search inside Lance handles the normalization on its end; if you want
to do offline math, normalize first (`mat / np.linalg.norm(mat, axis=1,
keepdims=True)`).

### Projection and filter (push-down to Lance)

```python
import os, lance
ds = lance.dataset(os.path.expanduser("~/.markq/chunks.lance"))
tbl = ds.scanner(
    columns=["path", "chunk_index", "text"],
    filter="collection = 'default'",
).to_table()
print(tbl.num_rows, "rows")
print(tbl.slice(0, 2))
```

```
8 rows
pyarrow.Table
path: string not null
chunk_index: int32 not null
text: string not null
----
path: [["/path/to/markq/README.md","/path/to/markq/README.md"]]
chunk_index: [[0,1]]
text: [["# markq\n\n`markq` (mark[down] + q[uery]) is a local-first Rust CLI ...",
        "`markq` (mark[down] + q[uery]) is a local-first Rust CLI for indexing a\nfolder of markdown and ans..."]]
```

The filter is parsed by Lance and pushed down — no full scan unless
necessary. Filter expression syntax is SQL-like (DataFusion).

## SQL via DataFusion

Pylance exposes DataFusion under `ds.sql(...)`:

```python
import os, lance
ds = lance.dataset(os.path.expanduser("~/.markq/chunks.lance"))
reader = (
    ds.sql("SELECT collection, COUNT(*) AS n FROM dataset GROUP BY 1")
      .build()
      .to_stream_reader()
)
print(reader.read_all())
```

```
pyarrow.Table
collection: string not null
n: int64 not null
----
collection: [["default"]]
n: [[8]]
```

In an `ds.sql(...)` query the dataset is bound as the table name `dataset`.
Rich joins, window functions, and aggregations work — this is the same
DataFusion the markq Rust crate uses internally.

## Time-travel — open an older version

Lance keeps every commit. `lance.dataset(uri, version=N)` opens a snapshot:

```python
import os, lance
uri = os.path.expanduser("~/.markq/chunks.lance")
for v in [1, 2, 3, 4]:
    ds_v = lance.dataset(uri, version=v)
    print(f"version={ds_v.version} rows={ds_v.count_rows()}")
```

```
version=1 rows=0
version=2 rows=0
version=3 rows=8
version=4 rows=8
```

Or list versions with timestamps:

```python
import os, lance
ds = lance.dataset(os.path.expanduser("~/.markq/chunks.lance"))
for v in ds.versions():
    print(v["version"], v["timestamp"], "rows=", v["metadata"].get("total_rows", "?"))
```

```
1 2026-05-05 10:09:30.480130 rows= 0
2 2026-05-05 10:09:30.481256 rows= 0
3 2026-05-13 19:19:28.746165 rows= 8
4 2026-05-13 19:19:28.759371 rows= 8
```

This is Lance's headline storage feature: zero-copy versioning of your data
without needing extra infrastructure.

## Statistics and storage layout

```python
import os, lance, json
ds = lance.dataset(os.path.expanduser("~/.markq/chunks.lance"))
print(json.dumps(ds.stats.dataset_stats(), indent=2, default=str))
```

```
{
  "num_deleted_rows": 0,
  "num_fragments": 1,
  "num_small_files": 1
}
```

These three numbers drive `markq compact`'s heuristic: when
`num_deleted_rows` and `num_small_files` cross a threshold relative to live
rows, run Lance compaction.

```python
print("fragments:", len(ds.get_fragments()))
print("indices: ", ds.list_indices())
print("tags:    ", ds.tags.list())
```

```
fragments: 1
indices:  [
  {'name': 'text_idx',      'type': 'Inverted',    'fields': ['text'],      'version': 3, ...},
  {'name': 'embedding_idx', 'type': 'IVF_HNSW_SQ', 'fields': ['embedding'], 'version': 6, ...},
]
tags:     {}
```

`text_idx` is the BM25 inverted index built by `markq index`.
`embedding_idx` is the `IvfHnswSq` vector index built by `markq embed`
(distance metric: Cosine).

## Vector search

After `markq embed` has run, `ds.scanner(nearest={...})` performs HNSW
KNN against the `embedding` column:

```python
import numpy as np, os, lance
ds = lance.dataset(os.path.expanduser("~/.markq/chunks.lance"))
# In practice you'd produce `query_vector` with the same embedder markq
# uses (Qwen3-Embedding-0.6B Q8_0); here we just use the first row's
# vector as a stand-in to demonstrate the API.
query_vector = ds.to_table(columns=["embedding"]).column("embedding")[0].as_py()
query_vector = np.asarray(query_vector, dtype=np.float32)

results = ds.scanner(
    nearest={
        "column": "embedding",
        "q": query_vector,        # numpy.ndarray, shape (1024,), dtype float32
        "k": 3,
        "metric": "cosine",
    },
    columns=["path", "chunk_index", "_distance"],
).to_table()
print(results)
```

Lance returns the `_distance` column alongside the requested ones;
cosine similarity is `1 - _distance`, which is what `markq vsearch`
prints. The Rust code path in `markq-index-lance` is the source of
truth; the Python API mirrors it for ad-hoc analytics.

## Downstream libraries that build on `lance.dataset(...)`

Anything that takes a `pyarrow.dataset.Dataset` works directly — pylance
datasets *are* `pyarrow.dataset.Dataset` instances (`isinstance(ds,
pyarrow.dataset.Dataset)` is `True`). Two notable consumers:

### Polars (lazy DataFrame, predicate / projection pushdown)

Run with `uv run --with pylance --with pyarrow --with polars python`.

```python
import os, lance, polars as pl
ds = lance.dataset(os.path.expanduser("~/.markq/chunks.lance"))
df = (
    pl.scan_pyarrow_dataset(ds)
      .filter(pl.col("collection") == "default")
      .select("path", "chunk_index", "text")
      .collect()
)
print(df.shape)
```

```
(8, 3)
```

Polars 1.40 has no `pl.read_lance`; the bridge is `scan_pyarrow_dataset(...)`.
The filter and column projection are pushed down into Lance — no full scan.

### DataFusion-Python (Rust SQL engine bound from Python)

Run with `uv run --with pylance --with datafusion python`. The
`register_dataset` path against a Lance dataset currently errors on
anything non-trivial — even a plain aggregate fails:

```python
import os, datafusion, lance
ctx = datafusion.SessionContext()
ctx.register_dataset("chunks", lance.dataset(os.path.expanduser("~/.markq/chunks.lance")))
ctx.sql("SELECT collection, COUNT(*) AS n FROM chunks GROUP BY 1").show()
```

```
DataFusion error: External error: TypeError: LanceFragment.scanner() takes 1 positional argument but 2 positional arguments (and 3 keyword-only arguments) were given
```

Upstream signature drift between `datafusion-python` and `pylance`. The
working pattern today is to materialize to Arrow first and register that
as a view:

```python
import os, datafusion, lance
ctx = datafusion.SessionContext()
ds = lance.dataset(os.path.expanduser("~/.markq/chunks.lance"))
ctx.register_view("chunks", ctx.from_arrow(ds.to_table()))
ctx.sql("SELECT collection, COUNT(*) AS n FROM chunks GROUP BY 1").show()
```

```
+------------+---+
| collection | n |
+------------+---+
| default    | 8 |
+------------+---+
```

There is **no off-the-shelf DataFusion CLI binary** that reads Lance —
upstream `datafusion-cli` 53.1.0 returns `Unable to find factory for LANCE`,
and no Lance-aware fork is published. Use the DuckDB CLI
([`usage/duckdb.md`](duckdb.md)) for shell SQL; use `datafusion-python` from
Python.

## What pylance *cannot* do that markq does

- Ingest, chunk, and embed markdown — that's the markq Rust binary's job.
- Run the cross-encoder reranker — markq's `llama-cpp-2` thread.
- RRF fusion of BM25 + vector ranked lists — done in markq-core.

pylance is for *reading* the dataset and for ML work that needs the rows in
NumPy. Building the dataset is markq's job.

## One-liner reference

```sh
# Schema
uv run --with pylance python -c "
import os, lance
print(lance.dataset(os.path.expanduser('~/.markq/chunks.lance')).schema)
"

# markq metadata
uv run --with pylance python -c "
import os, lance, json
print(json.dumps(lance.dataset(os.path.expanduser('~/.markq/chunks.lance')).config(), indent=2))
"

# Row count + version
uv run --with pylance python -c "
import os, lance
d = lance.dataset(os.path.expanduser('~/.markq/chunks.lance'))
print(f'version={d.version} rows={d.count_rows()}')
"

# All rows as a pandas DataFrame
uv run --with pylance --with pandas python -c "
import os, lance
df = lance.dataset(os.path.expanduser('~/.markq/chunks.lance')).to_table().to_pandas()
print(df.head())
"
```

## See also

- [`usage/markq.md`](markq.md) — the markq CLI itself.
- [`usage/duckdb.md`](duckdb.md) — same dataset accessed via SQL from the
  DuckDB CLI.
