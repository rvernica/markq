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

All output below is from real runs against the current dataset (zero rows;
real data lands once `markq index` is functional).

## Prerequisites

```sh
uv run --with pylance --with pyarrow python
# or, for ad-hoc snippets:
uv run --with pylance --with pyarrow python -c '...'
```

Pylance is currently `4.0.x`; markq pins lancedb-rust to `0.27.2`, which
embeds the same Lance file format (`2.0`).

## Open the dataset

```python
import lance
ds = lance.dataset("/home/vernica/.markq/chunks.lance")
print("uri:           ", ds.uri)
print("version:       ", ds.version)
print("latest_version:", ds.latest_version)
print("count_rows:    ", ds.count_rows())
```

```
uri:            /home/vernica/.markq/chunks.lance
version:        2
latest_version: 2
count_rows:     0
```

`version` increments on every write commit. The fresh dataset has two commits
— one for `create_table`, one for the `update_config` that wrote the markq
metadata.

## Read the markq metadata (the headline payoff over DuckDB)

```python
import lance, json
ds = lance.dataset("/home/vernica/.markq/chunks.lance")
print(json.dumps(ds.config(), indent=2))
```

```json
{
  "markq.lance_file_format_version": "2.0",
  "markq.lance_manifest_version": "1",
  "markq.lancedb_crate_version": "0.27.2",
  "markq.schema_version": "1",
  "lance.auto_cleanup.interval": "20",
  "lance.auto_cleanup.older_than": "14days"
}
```

This is what `markq doctor` will read via the Rust API and turn into
structured errors on mismatch. From Python you can sanity-check it without
going through the binary.

`config()` is a method (not a property) in pylance ≥ 4.0.

## Inspect the schema

```python
import lance
ds = lance.dataset("/home/vernica/.markq/chunks.lance")
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
import lance
ds = lance.dataset("/home/vernica/.markq/chunks.lance")
tbl = ds.to_table()
print(type(tbl).__name__)   # pyarrow.Table
print(tbl.num_rows, "rows,", tbl.num_columns, "columns")
```

```
Table
0 rows, 12 columns
```

### As a pandas DataFrame

```python
df = ds.to_table().to_pandas()
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
embedding         object   # NumPy array per row, shape (1024,)
context_id           str
schema_version     int32
```

The `embedding` column round-trips as Python objects (NumPy `float32` arrays
of length 1024); use `np.stack(df["embedding"])` for a 2D matrix once
embeddings are populated.

### Projection and filter (push-down to Lance)

```python
tbl = ds.scanner(
    columns=["path", "chunk_index", "text"],
    filter="collection = 'default'",
).to_table()
print(tbl)
```

```
pyarrow.Table
path: string not null
chunk_index: int32 not null
text: string not null
----
path: []
chunk_index: []
text: []
```

The filter is parsed by Lance and pushed down — no full scan unless
necessary. Filter expression syntax is SQL-like (DataFusion).

## SQL via DataFusion

Pylance exposes DataFusion under `ds.sql(...)`:

```python
import lance
ds = lance.dataset("/home/vernica/.markq/chunks.lance")
reader = (
    ds.sql("SELECT collection, COUNT(*) AS n FROM dataset GROUP BY 1")
      .build()
      .to_stream_reader()
)
print("schema:", reader.schema)
print(reader.read_all())
```

```
schema: collection: string not null
n: int64 not null
pyarrow.Table
collection: string not null
n: int64 not null
----
collection: []
n: []
```

In an `ds.sql(...)` query the dataset is bound as the table name `dataset`.
Rich joins, window functions, and aggregations work — this is the same
DataFusion the markq Rust crate uses internally.

## Time-travel — open an older version

Lance keeps every commit. `lance.dataset(uri, version=N)` opens a snapshot:

```python
import lance
for v in [1, 2]:
    ds_v = lance.dataset("/home/vernica/.markq/chunks.lance", version=v)
    print(f"version={ds_v.version} rows={ds_v.count_rows()}")
```

```
version=1 rows=0
version=2 rows=0
```

Or list versions with timestamps:

```python
ds = lance.dataset("/home/vernica/.markq/chunks.lance")
for v in ds.versions():
    print(v["version"], v["timestamp"], "rows=", v["metadata"]["total_rows"])
```

```
1 2026-05-05 10:09:30.480130 rows= 0
2 2026-05-05 10:09:30.481256 rows= 0
```

This is Lance's headline storage feature: zero-copy versioning of your data
without needing extra infrastructure.

## Statistics and storage layout

```python
import lance, json
ds = lance.dataset("/home/vernica/.markq/chunks.lance")
print(json.dumps(ds.stats.dataset_stats(), indent=2, default=str))
```

```
{
  "num_deleted_rows": 0,
  "num_fragments": 0,
  "num_small_files": 0
}
```

These three numbers drive `markq compact`'s heuristic: when
`num_deleted_rows` and `num_small_files` cross a threshold relative to live
rows, run Lance compaction.

```python
print("fragments:    ", ds.get_fragments())
print("indices:      ", ds.list_indices())
print("tags:         ", ds.tags.list())
```

```
fragments:     []
indices:       []
tags:          {}
```

`list_indices()` will populate once the FTS and HNSW vector indexes are
built.

## Vector search

The API shape, for reference — none of these run yet because the dataset
has no embeddings:

```python
# Once `markq embed` has populated the embedding column and built the index:
results = ds.scanner(
    nearest={
        "column": "embedding",
        "q": query_vector,        # numpy.ndarray, shape (1024,), dtype float32
        "k": 10,
        "metric": "cosine",
    },
    columns=["path", "chunk_index", "text"],
).to_table()
```

The Rust code path in `markq-index-lance` will be the source of truth; the
Python API mirrors it for ad-hoc analytics.

## Downstream libraries that build on `lance.dataset(...)`

Anything that takes a `pyarrow.dataset.Dataset` works directly — pylance
datasets *are* `pyarrow.dataset.Dataset` instances (`isinstance(ds,
pyarrow.dataset.Dataset)` is `True`). Two notable consumers:

### Polars (lazy DataFrame, predicate / projection pushdown)

```python
import lance, polars as pl
ds = lance.dataset("/home/vernica/.markq/chunks.lance")
df = (
    pl.scan_pyarrow_dataset(ds)
      .filter(pl.col("collection") == "default")
      .select("path", "chunk_index", "text")
      .collect()
)
print(df)
```

```
shape: (0, 3)
┌──────┬─────────────┬──────┐
│ path ┆ chunk_index ┆ text │
└──────┴─────────────┴──────┘
```

Polars 1.40 has no `pl.read_lance`; the bridge is `scan_pyarrow_dataset(...)`.
The filter and column projection are pushed down into Lance — no full scan.

### DataFusion-Python (Rust SQL engine bound from Python)

```python
import datafusion, lance
ctx = datafusion.SessionContext()
ctx.register_dataset("chunks", lance.dataset("/home/vernica/.markq/chunks.lance"))
ctx.sql("SELECT collection, COUNT(*) AS n FROM chunks GROUP BY 1").show()
```

```
+------------+---+
| collection | n |
+------------+---+
+------------+---+
```

Same DataFusion engine the Rust crate uses internally — `register_dataset`
hands it the Lance `TableProvider` from `lance-datafusion`. Aggregates and
`DESCRIBE` work; filter pushdown through this path currently errors
(`get_fragments() does not support filter yet` upstream), so for filtered
SQL materialize to Arrow first:

```python
ctx.register_view("chunks_arrow", ctx.from_arrow(ds.to_table()))
ctx.sql("SELECT path FROM chunks_arrow WHERE collection = 'default'").show()
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
import lance
print(lance.dataset('/home/vernica/.markq/chunks.lance').schema)
"

# markq metadata
uv run --with pylance python -c "
import lance, json
print(json.dumps(lance.dataset('/home/vernica/.markq/chunks.lance').config(), indent=2))
"

# Row count + version
uv run --with pylance python -c "
import lance
d = lance.dataset('/home/vernica/.markq/chunks.lance')
print(f'version={d.version} rows={d.count_rows()}')
"

# All rows as a pandas DataFrame
uv run --with pylance --with pandas python -c "
import lance
df = lance.dataset('/home/vernica/.markq/chunks.lance').to_table().to_pandas()
print(df.head())
"
```

## See also

- [`usage/markq.md`](markq.md) — the markq CLI itself.
- [`usage/duckdb.md`](duckdb.md) — same dataset accessed via SQL from the
  DuckDB CLI.
