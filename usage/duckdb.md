# Querying markq's Lance dataset from DuckDB

The markq dataset at `~/.markq/chunks.lance` is a bare Lance directory, so you
can run SQL against it from DuckDB without going through the `markq` binary.
This is the composable-stack payoff that comes from storing chunks as a bare
Lance dataset rather than an opaque internal index.

All output below is from real runs against the current dataset (zero rows;
schema and metadata are in place, real chunks land once `markq index`
becomes functional).

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
`chunks` table:

```sql
LOAD lance;
ATTACH '/home/vernica/.markq' AS m (TYPE lance);
SHOW TABLES FROM m;
```

```
┌─────────┐
│  name   │
├─────────┤
│ chunks  │
└─────────┘
```

Once attached, `m.chunks` behaves like a regular DuckDB table.

### Inspect the schema

```sql
DESCRIBE m.chunks;
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
natively, no UDF needed.

### Count and select

```sql
SELECT COUNT(*) AS rows FROM m.chunks;
```

```
┌───────┐
│ rows  │
│ int64 │
├───────┤
│     0 │
└───────┘
```

```sql
SELECT collection, path, chunk_index FROM m.chunks LIMIT 5;
```

```
┌────────────┬─────────┬─────────────┐
│ collection │  path   │ chunk_index │
└────────────┴─────────┴─────────────┘
        0 rows
```

(Once `markq index` is functional, this query returns real rows.)

## Function-style scan (alternative)

If you'd rather not `ATTACH`, `__lance_scan(<dataset path>)` reads a single
dataset directly. The double underscore is the extension's "internal API"
marker — it's the function older Lance docs call `lance_scan`; the
published extension renames it.

```sql
LOAD lance;
SELECT id, path, chunk_index
FROM __lance_scan('/home/vernica/.markq/chunks.lance')
LIMIT 5;
```

```
┌─────────┬─────────┬─────────────┐
│   id    │  path   │ chunk_index │
└─────────┴─────────┴─────────────┘
       0 rows
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

Expected shapes:
- **BM25**: `SELECT * FROM lance_fts('m.chunks', 'text', 'query', false, 10);`
- **Vector**: `SELECT * FROM lance_vector_search('m.chunks', 'embedding', [0.1, ...], ...);`
- **Hybrid**: `lance_hybrid_search(...)` with `alpha` weighting BM25 vs vector.

The exact argument shapes will be confirmed against real data once the index
is built; for now the function existence is what matters.

## Lance auto-cleanup config

LanceDB tracks dataset versions and auto-cleans old ones. The current
settings live in the dataset config and are visible via:

```sql
LOAD lance;
SELECT * FROM __lance_show_auto_cleanup('/home/vernica/.markq/chunks.lance');
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
`markq.lancedb_crate_version`) live in Lance's user `config` map. The DuckDB
extension does not surface this map through SQL — only its own
`auto_cleanup.*` keys are exposed.

To read the markq metadata, use pylance instead:

```sh
uv run --with pylance python -c "
import lance, json
print(json.dumps(lance.dataset('/home/vernica/.markq/chunks.lance').config(), indent=2))
"
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

`markq doctor` will read these directly via the Rust API and turn
mismatches into structured errors. DuckDB is for ad-hoc data exploration.

## One-liner reference

```sh
# Open an interactive DuckDB shell with markq attached.
~/.duckdb/cli/1.5.0/duckdb -c "
LOAD lance;
ATTACH '/home/vernica/.markq' AS m (TYPE lance);
" -interactive

# Count chunks per collection.
~/.duckdb/cli/1.5.0/duckdb -c "
LOAD lance;
ATTACH '/home/vernica/.markq' AS m (TYPE lance);
SELECT collection, COUNT(*) FROM m.chunks GROUP BY 1 ORDER BY 2 DESC;
"

# Find the longest unembedded chunks.
~/.duckdb/cli/1.5.0/duckdb -c "
LOAD lance;
ATTACH '/home/vernica/.markq' AS m (TYPE lance);
SELECT path, chunk_index, length(text) AS chars
FROM m.chunks
WHERE embedding IS NULL
ORDER BY chars DESC
LIMIT 10;
"
```

## See also

- [`usage/markq.md`](markq.md) — the markq CLI itself.
- [`usage/pylance.md`](pylance.md) — same dataset from Python (covers
  polars and datafusion-python as well).
