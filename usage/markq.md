# Using the `markq` CLI

This file covers what the binary actually does today. The current build
ships the workspace skeleton, `markq inspect`, `markq index` (markdown
walk + chunk + FTS index build), `markq search` (BM25 over the FTS
index), `markq embed` (fills the embedding column via Qwen3-Embedding-
0.6B Q8_0 over llama.cpp), `markq vsearch` (cosine KNN over the HNSW
vector index), `markq query` (hybrid BM25 + vector retrieval fused
with weighted Reciprocal Rank Fusion), and `markq rerank` (cross-encoder
rerank of piped candidates via Qwen3-Reranker-0.6B Q8_0, also available
as `query --rerank` for an integrated post-fusion pass). Every other v1
subcommand is registered (so `markq --help` shows the final surface) but
stubs out at runtime. Each will be lit up as the corresponding feature
work lands.

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
  index        Index a path into the default collection
  embed        Generate embeddings for unembedded rows
  embed-query  Embed one query string and print the vector as JSON
  collection   Manage collections
  context      Manage the context tree
  search       BM25 retrieval
  vsearch      Vector retrieval
  query        Hybrid retrieval (BM25 + vector + RRF)
  rerank       Hybrid retrieval + cross-encoder rerank
  get          Fetch one document by path or `#docid`
  multi-get    Fetch many documents by glob/csv/`#ids`
  compact      Reclaim space from tombstoned chunks
  doctor       Diagnose index, model, and dimension issues
  models       Manage the GGUF model cache
  watch        Filesystem watch + incremental reindex (`--features watch`)
  serve        Run the MCP server over stdio
  inspect      Print the dataset path, Arrow schema, row count, and recorded metadata
  status       Show index health, collection sizes, model state
  config       Show or edit the markq config
  help         Print this message or the help of the given subcommand(s)
```

Per-subcommand help works even on stubs — useful for previewing the final
flag surface:

```sh
markq get --help
```

```
Fetch one document by path or `#docid`

Usage: markq get [OPTIONS] <TARGET>

Arguments:
  <TARGET>

Options:
      --dataset <DATASET>  Path to the chunk dataset. Defaults to `~/.markq/chunks.lance`
      --full
  -h, --help               Print help
```

Calling a stub fails fast with exit 1:

```sh
markq get "anything"
```

```
Error: not implemented yet
```

## What works today

| Command | Status |
|---------|--------|
| `inspect` | ✅ Implemented |
| `index <path>` | ✅ Implemented (walk, chunk, FTS build) |
| `search <query>` | ✅ Implemented (BM25 via Lance inverted index) |
| `embed` | ✅ Implemented (Qwen3-Embedding-0.6B Q8_0, HNSW index build) |
| `vsearch <query>` | ✅ Implemented (cosine KNN over the HNSW vector index) |
| `query <query>` | ✅ Implemented (BM25 + vector + weighted RRF; `--explain` available; `--rerank` for an integrated cross-encoder pass) |
| `embed-query <query>` | ✅ Implemented (prints the query embedding as JSON for external vector search) |
| `rerank` | ✅ Implemented (cross-encoder rerank of stdin candidates via Qwen3-Reranker-0.6B Q8_0) |
| `get`, `multi-get`, `compact`, `doctor` | Stub |
| `status`, `config` | Stub |
| `collection {add,list,remove}` | Stub |
| `context add`, `models {pull,ls}` | Stub |
| `watch`, `serve` | Stub |
| `search --explain`, `search --collection <name>` | Returns a structured "not implemented" error |
| `vsearch --explain`, `vsearch --collection <name>` | Same — recognized but gated |
| `query --collection <name>` | Recognized but gated |

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

After `markq embed` has run on the dataset, two more lines appear at
the bottom:

```
embedder_model:            Qwen/Qwen3-Embedding-0.6B-GGUF/Q8_0
embedder_dim:              1024
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

Passing `--collection` (`-c`) `<name>` is parsed but not yet enforced — everything currently
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
Default `--top-k` (`-n`) is 10. Empty result sets print `(no results)`:

```sh
markq search "nonexistent_term_xyz"
```

```
(no results)
```

### Useful flags

- `--top-k` (`-n`) `<K>` — top-K (default 10).
- `--min-score <F>` — filter results below the BM25 score threshold.
- `--files` — print just the file paths (one per line, deduped).
- `--json` — emit the full result rows as a JSON array.
- `--all` — search across all collections (currently a no-op since only
  `default` exists).
- `--explain` and `--collection` (`-c`) `<name>` — recognized but stubbed; exit with
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

## `markq embed`

`embed` fills the `embedding` column for every row where it's currently
NULL. The first run downloads the embedder GGUF from Hugging Face
(`Qwen/Qwen3-Embedding-0.6B-GGUF`, ~640 MB, Q8_0 quantization) into
`~/.cache/markq/models/`; subsequent runs reuse the cached file.

```sh
markq embed
```

```
embedded 13 row(s) over 1 batch(es) (model=Qwen/Qwen3-Embedding-0.6B-GGUF/Q8_0, dim=1024)
```

What runs under the hood:

1. `model_cache::ensure_model` — checks `~/.cache/markq/models/` for the
   GGUF, downloads via `hf-hub` if absent. Override the cache root with
   `MARKQ_MODELS_DIR=/some/path`.
2. `Embedder::load` — spawns one owner thread that holds the
   `LlamaModel` + `LlamaContext` (the context is `!Send`, so a single
   thread is the only correct shape). Requests flow in over a bounded
   `crossbeam-channel`.
3. `validate_or_record_embedder` — on first embed, writes
   `markq.embedder_model` and `markq.embedder_dim` to the dataset
   metadata. On later runs, mismatches raise a typed
   `EmbedderDimMismatch` error rather than silently corrupting
   recall.
4. Each unembedded row is tokenized, decoded with pooling=`Last` (the
   model's documented default), and merge-inserted back keyed on `id`.
5. After each batch, an `IvfHnswSq` Cosine vector index is
   (re)built on the `embedding` column.

`embed` is idempotent — running it on a fully-embedded dataset is a
no-op:

```sh
markq embed
```

```
embedded 0 row(s) over 0 batch(es) (model=Qwen/Qwen3-Embedding-0.6B-GGUF/Q8_0, dim=1024)
```

Ctrl-C drains cleanly: the in-flight batch finishes (decoding is never
aborted mid-batch), the result is flushed, and the process exits. No
partial-batch corruption.

**What about the vector and FTS indexes?** Both are rebuilt once at the end
of the run, not per batch. A graceful Ctrl-C still runs that rebuild over
whatever was embedded so far, so the indexes stay consistent with the data.
A *hard* kill (SIGKILL, crash, power loss) mid-run skips the rebuild:
embedded rows are still durable, but because each batch's merge-insert
rewrites row fragments, the FTS index is left stale (BM25 degrades) and the
vector index is absent or stale. Re-running `markq embed` embeds the
remaining rows and rebuilds both indexes, restoring consistency.

The embedder thread holds the model in RAM only for the lifetime of the
`markq embed` invocation. Re-runs reload the weights, but llama.cpp
memory-maps the GGUF and the file's pages stay in the OS page cache,
so the second cold-start is sub-second. A long-running daemon mode
(weights stay warm across queries) lands with `markq serve`.

GPU offload is opt-in via Cargo features:

```sh
cargo build --release --features vulkan   # or cuda
```

Without those features the binary stays CPU-only and ignores GPU
detection. The current build supports the inference-thread + bounded-
channel pattern that a future reranker and HyDE generator will copy
verbatim.

## `markq vsearch`

Cosine vector retrieval. Embeds the query with the same model recorded
in the dataset's `markq.embedder_model` key, then runs HNSW KNN against
the `embedding` column.

```sh
markq vsearch "how does reranking work" -n 3
```

```
  1.   0.094  markq://default/SYNTAX.md#6
     Vec queries are natural language questions. No special syntax — just write what you're looking for. …
  2.  -0.014  markq://default/SYNTAX.md#7
     Hyde queries are hypothetical answer passages (50-100 words). Write what you expect the answer to lo…
  3.  -0.103  markq://default/SYNTAX.md#9
     An expand query stands alone; it's not mixed with typed lines. You can either rely on the default un…
```

The score column is the **cosine similarity** (`1 - distance`); higher
is better, matching the BM25 convention. Values sit in `[-1, 1]` —
unlike BM25 scores which are unbounded log-domain numbers.

`--json`, `--files`, `--top-k` (`-n`), `--min-score`, and `--all` all work the
same as `markq search` (the formatters are shared). `--explain` is
recognized but gated on `vsearch` — for explained retrieval, use
`markq query --explain`. `--collection` (`-c`) is gated until the
multi-collection wiring lands.

Running `vsearch` against a dataset without embeddings produces a clean
error and does **not** load the model:

```sh
markq --dataset /tmp/empty.lance index README.md
markq --dataset /tmp/empty.lance vsearch "anything"
```

```
Error: no embeddings in this dataset; run `markq embed` first to populate them
```

Running `vsearch` against a path that doesn't exist at all is a separate
error. A read-only command won't create a dataset on the fly, so it fails
rather than silently returning nothing:

```
Error: dataset not found at /tmp/missing.lance (run `markq index <path>` first)
```

Same shape if the dataset was embedded with a model this binary doesn't
know about:

```
Error: dataset was built with embedder some-other-model/Q4_K_M, but this build only knows Qwen/Qwen3-Embedding-0.6B-GGUF/Q8_0
```

## `markq query`

Hybrid retrieval. Runs BM25 and vector KNN concurrently against the same
dataset and fuses the two ranked lists with weighted Reciprocal Rank
Fusion (RRF). Reuses the embedder recorded in
`markq.embedder_model` — same model-validation guard as `vsearch`.

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
      --files
      --all
  -n, --top-k <TOP_K>            [default: 10]
      --min-score <MIN_SCORE>
      --explain                  Per-stage timing + RRF contribution trace
      --rerank                   Cross-encoder rerank of the fused candidate pool (query only). Results are then ordered by cross-encoder relevance instead of the fusion score, and each hit's score becomes that relevance probability in [0, 1]; `--min-score` (if given) filters on this probability, not the fusion score. Only the top 64 fused candidates are reranked, so combining this with a larger `--top-k` (`-n`) or `--all` is capped at 64 results
```

```sh
markq query "how does reranking work" -n 5
```

```
  1.   0.072  markq://default/SYNTAX.md#6
     Vec queries are natural language questions. No special syntax — just write what you're looking for. …
  2.   0.071  markq://default/SYNTAX.md#2
     | Type | Method | Description | |------|--------|-------------| | `lex` | BM25 | Keyword search with…
  3.   0.041  markq://default/SYNTAX.md#7
     Hyde queries are hypothetical answer passages (50-100 words). Write what you expect the answer to lo…
  4.   0.041  markq://default/SYNTAX.md#5
     ``` lex: CAP theorem consistency lex: "machine learning" -"deep learning" lex: auth -oauth -saml ```…
  5.   0.041  markq://default/SYNTAX.md#10
     - At most one `intent:` line per query document - `intent:` cannot appear alone — at least one `lex:…
```

The score column is the **fused RRF score**, not BM25 or cosine. It is
the weighted sum of `weight / (k + rank)` over the two source lists,
plus a small bonus for ranks 1–3. Defaults: `k=60`, `weight_lex=0.75`,
`weight_vec=0.60`, top-3 bonus `[+0.05, +0.02, +0.02]`. These constants
are not currently exposed as CLI flags — tuning lives in
`FusionConfig::default()` in `markq-core`. Absolute values have no
meaning across queries — only the within-query ordering does.

`--json`, `--files`, `--top-k` (`-n`), `--min-score`, and `--all` behave identically
to `markq search` and `markq vsearch`.

### `--explain`

`--explain` writes a per-stage timing summary plus a contribution table
to **stderr**, leaving stdout free for piping (`--files | xargs`,
`--json | jq`):

```sh
markq query "how does reranking work" --explain -n 5 2> trace.txt
cat trace.txt
```

```
bm25:   10 hits in 20ms
embed:  query in 120ms
vector: 13 hits in 12ms
fuse:   13 unique docs in 0ms

rank  id               final      lex(rank,w)      vec(rank,w)   bonus
1     d0bd2b3da7b7    0.0716       ( 4, 0.75)       ( 1, 0.60)    0.05
2     5affe2beb176    0.0711       ( 1, 0.75)       ( 8, 0.60)    0.05
3     720f9224cbc3    0.0412       ( 5, 0.75)       ( 2, 0.60)    0.02
4     ed131760dd53    0.0410       ( 3, 0.75)       ( 6, 0.60)    0.02
5     293d42518fc3    0.0406       ( 8, 0.75)       ( 3, 0.60)    0.02
```

`bm25` and `embed` are each timed inside their own future and the two are
issued concurrently via `tokio::join!`, so on a multi-core machine the
wall-clock cost is roughly `max(bm25, embed)` even though each line
reports its own duration. `vector` runs after the embedder produces a
vector. `fuse` runs in-process and is typically sub-millisecond. `lex(rank,w)` is the document's 1-based rank in the
BM25 list and the configured weight; missing means the document didn't
appear there (`(  - ,   - )`). Same shape for `vec`. `bonus` is the
top-rank bonus contribution (0 if the document was ranked 4+ in every
list).

The fetch depth before fusion is `max(2 × n, 20)`, where `n` is the
requested `--top-k` (`-n`) — each side is over-fetched so the fused top-k
has real candidates when the two lists disagree near the head, with a
floor of 20 for very small `n`.

A query against a dataset without embeddings fails the same way
`vsearch` does, with no model load:

```
Error: no embeddings in this dataset; run `markq embed` first to populate them
```

### `--rerank`

`--rerank` adds a cross-encoder pass **after** fusion: it takes the fused
candidate pool, re-scores every candidate against the query with the
same Qwen3-Reranker-0.6B Q8_0 model that backs standalone `markq rerank`,
and reorders the output by that cross-encoder relevance score instead of
the fusion score. Fusion itself is unchanged — `--rerank` only affects
what happens to the fused list before it's printed.

```sh
markq query "how does rank fusion work" --rerank --top-k 5
```

```
  1.   1.000  markq://default/rank-fusion.md#0
     # Rank Fusion in Hybrid Search Rank fusion, most commonly implemented as Reciprocal Rank Fusion (RRF…
  2.   1.000  markq://default/rank-fusion.md#1
     Rank fusion, most commonly implemented as Reciprocal Rank Fusion (RRF), combines result lists from m…
  3.   0.013  markq://default/reranking.md#0
     # Cross-Encoder Reranking A cross-encoder reranker jointly encodes a query and a candidate passage i…
  4.   0.001  markq://default/markdown-chunking.md#0
     # Chunking Markdown Documents Splitting long markdown files into smaller chunks improves retrieval q…
```

(The demo corpus above has only 4 chunks total, so `--top-k 5` naturally
returns 4 results — this is corpus-size behavior, not a `--rerank`
quirk.)

A few things worth calling out:

- `-n`/`--top-k` is a single flag with two names (`-n` is the short
  alias) — it means "keep this many" whether or not `--rerank` is set.
- With `--rerank`, only the **top 64** fused candidates are ever sent to
  the cross-encoder (the fan-in cap that keeps a single reranked query
  bounded in latency). Passing a larger `--top-k` (`-n`), or `--all`,
  still caps the reranked result set at 64 — the cross-encoder simply
  never sees candidates ranked below 64 in the fused list.
- `--min-score` changes meaning under `--rerank`: normally it filters on
  the raw fusion score (an unbounded RRF value), but with `--rerank` set
  it filters on the cross-encoder relevance score instead, which lives
  in `[0, 1]`. A `--min-score` tuned for fusion scores will not mean the
  same thing once `--rerank` is added, and vice versa.
- `--explain`'s per-stage trace gains a `rerank` timing line when
  `--rerank` is set, alongside the existing `bm25` / `embed` / `vector`
  / `fuse` lines.

## `markq rerank`

Cross-encoder re-scoring of a candidate list produced by another
command. `rerank` reads a JSON array of candidates from **stdin** — the
same shape as `search`/`vsearch`/`query`'s `--json` output — scores each
one against `--query` with the Qwen3-Reranker-0.6B Q8_0 model, and
prints them best-first. Unlike `search`/`vsearch`/`query`, `rerank`
takes no `<QUERY>` positional; the query is always `--query <TEXT>`,
since the query used for retrieval and the query used for reranking are
allowed to differ.

```sh
markq rerank --help
```

```
Cross-encoder rerank of stdin candidates

Usage: markq rerank [OPTIONS] --query <QUERY>

Options:
      --dataset <DATASET>          Path to the chunk dataset. Defaults to `~/.markq/chunks.lance`
      --query <QUERY>              The query to score every stdin candidate against
      --top-k <TOP_K>              Keep only the top `k` candidates after reordering
      --json
      --instruction <INSTRUCTION>  Override the default retrieval instruction sent to the reranker
  -h, --help                       Print help
```

- `--query <TEXT>` (required) — the query every stdin candidate is
  scored against.
- `--top-k <N>` — keep only the N highest-scoring candidates; default is
  all candidates, reordered.
- `--json` — emit a JSON array of `RerankedCandidate` instead of the
  human ranked list.
- `--instruction <TEXT>` — overrides the built-in retrieval instruction
  template sent to the reranker (advanced use; the default instruction
  is tuned for retrieval-style queries).

Piping a first-stage `query --json` result straight into `rerank` is the
standalone flow (equivalent in effect to `query --rerank`, but as two
separate invocations — useful when the query used to retrieve differs
from the query used to rerank, or when reranking candidates gathered
from elsewhere):

```sh
markq query "how does rank fusion work" --json \
  | markq rerank --query "how does rank fusion work" --top-k 5 --json
```

```json
[
  {
    "id": "77c1b2ebf88f14c5f01bdfb698d2a4305ac5fc7eec138a6a1757592c305473d8",
    "text": "# Rank Fusion in Hybrid Search\n\nRank fusion, most commonly implemented as Reciprocal Rank Fusion (RRF), combines\nresult lists from multiple retrieval systems — such as a lexical BM25 search and\na dense vector similarity search — into a single merged ranking. Each document's\nfused score is the sum of `1 / (k + rank)` across all the lists it appears in,\nwhere `k` is a smoothing constant (commonly 60). Rank fusion works well because\nit only needs the rank position from each retriever, not calibrated scores, so\nit sidesteps the problem of BM25 and cosine-similarity scores living on\nincompatible scales.\n",
    "rerank_score": 0.9999784,
    "rank": 1,
    "chunk_index": 0,
    "path": "/home/user/rerank-demo/corpus/rank-fusion.md",
    "score": 0.12213115394115448,
    "uri": "markq://default/rank-fusion.md"
  },
  {
    "id": "dadeaba4fc572d9fb9fe6945267998ed28d50a53c95546f3b00afe949d507c40",
    "text": "Rank fusion, most commonly implemented as Reciprocal Rank Fusion (RRF), combines\nresult lists from multiple retrieval systems — such as a lexical BM25 search and\na dense vector similarity search — into a single merged ranking. Each document's\nfused score is the sum of `1 / (k + rank)` across all the lists it appears in,\nwhere `k` is a smoothing constant (commonly 60). Rank fusion works well because\nit only needs the rank position from each retriever, not calibrated scores, so\nit sidesteps the problem of BM25 and cosine-similarity scores living on\nincompatible scales.\n\n## Why hybrid retrieval needs fusion\n\nA pure lexical search misses paraphrases and synonyms. A pure vector search can\ndrift away from exact keyword matches. Rank fusion is the glue that lets a\nhybrid system get the best of both: the precision of exact term matches and the\nrecall of semantic similarity.\n",
    "rerank_score": 0.9998179,
    "rank": 2,
    "chunk_index": 1,
    "path": "/home/user/rerank-demo/corpus/rank-fusion.md",
    "score": 0.06177419424057007,
    "uri": "markq://default/rank-fusion.md"
  },
  {
    "id": "b447669e87cc15a7c4046b7d48c6d7023ccd2c2ed329eace5efefcf881152d98",
    "text": "# Cross-Encoder Reranking\n\nA cross-encoder reranker jointly encodes a query and a candidate passage in a\nsingle forward pass, producing a relevance score that is typically far more\naccurate than the first-stage bi-encoder or lexical retrieval score. Rerankers\nare usually applied to only the top few dozen candidates from a first-stage\nretriever, since scoring every document in a corpus with a cross-encoder is too\nslow for large collections.\n",
    "rerank_score": 0.012855237,
    "rank": 3,
    "chunk_index": 0,
    "path": "/home/user/rerank-demo/corpus/reranking.md",
    "score": 0.029523808509111404,
    "uri": "markq://default/reranking.md"
  },
  {
    "id": "87ea1d0c13899835b1aa142a4613173f6d00523dd0a340024671e37b9a8491bb",
    "text": "# Chunking Markdown Documents\n\nSplitting long markdown files into smaller chunks improves retrieval quality.\nA chunker typically respects heading boundaries and keeps code blocks intact,\nproducing chunks of a few hundred to about a thousand tokens each. Overlapping\nchunk boundaries slightly can help preserve context across a chunk split.\n",
    "rerank_score": 0.0006887511,
    "rank": 4,
    "chunk_index": 0,
    "path": "/home/user/rerank-demo/corpus/markdown-chunking.md",
    "score": 0.00937500037252903,
    "uri": "markq://default/markdown-chunking.md"
  }
]
```

The `RerankedCandidate` shape adds `rerank_score` (the cross-encoder
relevance score in `[0, 1]`) and `rank` (1-based position in the
reranked output) to each object, while passing through every other
field the input candidate carried (`id`, `text`, `path`, `uri`,
`chunk_index`, the original `score`, ...) — nothing from the first
stage is dropped. Ordering is best-first by `rerank_score`; a fixed
model + quantization gives byte-identical output ordering for identical
input, run after run, with ties breaking by input order.

The default (non-`--json`) output is the same ranked-list shape as
`search`/`vsearch`/`query`, just with the cross-encoder score in the
score column:

```sh
echo '[{"id":"a","text":"only one"}]' | markq rerank --query "x"
```

```
  1.   0.013  a
     only one
```

### Edge cases

An empty candidate array on stdin short-circuits before the reranker
model is even loaded — no llama.cpp log noise, empty stdout, exit 0:

```sh
echo '[]' | markq rerank --query "x"
echo "exit=$?"
```

```
exit=0
```

A single candidate is scored and returned, not dropped (shown above). A
candidate missing `id` or `text` — or malformed input JSON altogether —
fails fast with an actionable stderr message rather than skipping the
bad entry silently.

## `markq embed-query`

Embeds a single query string with the canonical embedder and prints the
vector as a one-line JSON array on stdout — nothing else. This lets
external tools run their own vector search against the markq dataset
without loading a GGUF themselves: splice the array into DuckDB's
`lance_vector_search`, feed it to pylance, or pipe it through `jq`.

```sh
markq embed-query "how does retrieval work"
```

```
[0.0123,-0.0456,0.0789, ... ]
```

It uses the exact same `Embedder::load` + `embedder.embed` path as
`vsearch` and `query`, so the vector is cosine-compatible with the
dataset's stored embeddings by construction. Before loading the model it
validates the dataset's recorded `markq.embedder_model` — if the dataset
was built with a different embedder, or has no embeddings yet, it fails
loudly with the same message as `vsearch` rather than printing a vector
that would silently mismatch. See [`usage/duckdb.md`](duckdb.md) for the
end-to-end SQL composition.

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
