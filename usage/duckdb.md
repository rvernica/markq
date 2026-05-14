# Querying markq's Lance dataset from DuckDB

The markq dataset at `~/.markq/chunks.lance` is a bare Lance directory, so you
can run SQL against it from DuckDB without going through the `markq` binary.
This is the composable-stack payoff that comes from storing chunks as a bare
Lance dataset rather than an opaque internal index.

All output below is from real runs against the default dataset after
indexing this repo's `README.md` (8 chunks, no embeddings yet — those
populate once `markq embed` is functional). Reproduce it with:

```sh
markq index README.md
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
natively, no UDF needed. It stays NULL until `markq embed` is functional.

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

**Vector** (`lance_vector_search`) and **Hybrid** (`lance_hybrid_search`,
`alpha` weighting BM25 vs vector) need populated embeddings and an HNSW
index first; they'll light up once `markq embed` is functional.

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
`markq.lancedb_crate_version`) live in Lance's user `config` map. The DuckDB
extension does not surface this map through SQL — only its own
`auto_cleanup.*` keys are exposed.

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
  "markq.lance_manifest_version": "1",
  "markq.schema_version": "1",
  "markq.lancedb_crate_version": "0.27.2",
  "lance.auto_cleanup.interval": "20",
  "markq.lance_file_format_version": "2.0"
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

## See also

- [`usage/markq.md`](markq.md) — the markq CLI itself.
- [`usage/pylance.md`](pylance.md) — same dataset from Python (covers
  polars and datafusion-python as well).
