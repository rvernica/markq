# markq demo: BM25 indexer, embedder, vector + hybrid retrieval

A walkthrough across what's working today. Part 1 covers the Best
Matching 25 (BM25) path: an index, full-text search (FTS), and
`markq search`. Part 2 covers the vector path: `markq embed`, a
Hierarchical Navigable Small World (HNSW) index, and `markq vsearch`.
Part 3 covers hybrid `markq query`, which fuses BM25 and vector
retrieval with Reciprocal Rank Fusion (RRF). Part 4 covers the
cross-encoder reranker — `markq rerank` and `query --rerank` — which
re-scores the top candidates for precision.

All output below is captured from the real binary against a snapshot of
the [GitHub Docs](https://github.com/github/docs) "get started" section
— 118 markdown files, 740 chunks. Re-run the commands and the output
should match modulo absolute paths and score values.

## Setup

The corpus is a sparse, shallow checkout of one subtree of the
`github/docs` repo, so it lands as plain `*.md` without pulling the
whole repository:

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/github/docs.git /tmp/gh-docs
git -C /tmp/gh-docs sparse-checkout set content/get-started
```

These are real authored docs, not a synthetic fixture: each file carries
YAML frontmatter and GitHub's Liquid templating (`{% data variables… %}`,
`{% ifversion … %}`). markq indexes that markup as literal text, so it
occasionally shows up in chunk previews below — that's the corpus, warts
and all, not a rendering bug.

```sh
cargo build --release -p markq-cli                    # CPU build, works everywhere
cargo build --release -p markq-cli --features vulkan  # optional: GPU offload via Vulkan
DS=/tmp/markq-demo/chunks.lance
mq() { ./target/release/markq --dataset "$DS" "$@"; }
```

`--dataset` overrides the default `~/.markq/chunks.lance` so the demo
runs against a throwaway dataset and leaves the user's index alone. The
`--features vulkan` build accelerates every model stage — `embed`,
`vsearch`, `query`, and `rerank` — on a GPU (the device-selection log is
shown in [§6](#6-embed-the-corpus--markq-embed)); the plain build runs the
same model on CPU and produces identical embeddings, just slower.

`--version` reports which build you have — the compiled-in inference
backend is appended in parentheses, so there's no need to inspect the
binary:

```sh
./target/release/markq --version
```

```
markq 0.1.0 (backend: vulkan)
```

A plain CPU build prints `(backend: cpu)`. Note this is the backend
*compiled in*; a `vulkan` build still falls back to CPU at runtime if no
usable Vulkan driver is present on the host.

## 1. Index a markdown tree

```sh
mq index /tmp/gh-docs/content/get-started
```

```
indexed 118 file(s), 740 chunk(s) (0 skipped, 0 removed) into /tmp/markq-demo/chunks.lance
```

The indexer walks the path for `*.md` files, computes a blake3
`content_hash`, chunks each file via `markq-chunker` (~900 token target,
~15% overlap, heading-boundary preferred), and writes one Lance row per
chunk. `tokens` is filled in at index time with each chunk's token
count; `embedding` is left null until `markq embed` runs, and
`context_id` is reserved (always null for now) for the context tree.

Indexing is incremental: re-running it diffs each file's `content_hash`
against what's already stored, so unchanged files are skipped (keeping
their existing rows and embeddings), edited files have their old chunks
replaced, new files are added, and files deleted from disk are pruned.
Re-running on an unchanged tree is a no-op:

```sh
mq index /tmp/gh-docs/content/get-started
```

```
indexed 0 file(s), 0 chunk(s) (118 skipped, 0 removed) into /tmp/markq-demo/chunks.lance
```

## 2. Inspect the dataset

```sh
mq inspect
```

```
dataset path:  /tmp/markq-demo/chunks.lance
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
rows:          740
schema_version:            1
lance_manifest_version:    1
lance_file_format_version: 2.0
lancedb_crate_version:     0.27.2
```

The schema already reserves the `collection`, `uri`, and `context_id`
columns that multi-collection routing and the context tree will use —
present but unused today, so those features can land as plain code rather
than a Lance schema migration and full re-embed. (`schema_version` is `1`;
reserving the columns up front is exactly what lets it stay at `1` when
those features ship, instead of bumping to a migrated version.)

## 3. BM25 search — fast keyword retrieval

```sh
mq search "pull request review" -n 5
```

```
  1.   9.666  markq://default/using-github/github-flow.md#6
     If your repository has checks configured to run on pull requests, you will see any checks that faile…
  2.   9.576  markq://default/using-github/github-flow.md#7
     Reviewers should leave questions, comments, and suggestions. Reviewers can comment on the whole pull…
  3.   9.366  markq://default/start-your-journey/hello-world.md#7
     1. Click the **Pull requests** tab of your `hello-world` repository. 1. Click **New pull request**. …
  4.   9.028  markq://default/using-github/github-flow.md#5
     Continue to make, commit, and push changes to your branch until you are ready to ask for feedback. >…
  5.   8.771  markq://default/start-your-journey/hello-world.md#8
     We won't cover reviewing pull requests in this tutorial, but if you're interested in learning more, …
```

The score column is LanceDB's BM25 score (higher = better; not
normalized to [0, 1]). The URI uses the reserved `markq://<collection>/`
scheme so the rendering is stable when multi-collection lands.

## 4. Output formats — for agents and pipelines

`--files` deduplicates by source path, in first-seen rank order. This
is the agent-friendly form for `xargs cat`:

```sh
mq search "two-factor authentication" --files
```

```
/tmp/gh-docs/content/get-started/onboarding/getting-started-with-your-github-account.md
/tmp/gh-docs/content/get-started/using-github/github-mobile.md
/tmp/gh-docs/content/get-started/onboarding/getting-started-with-github-team.md
/tmp/gh-docs/content/get-started/onboarding/getting-started-with-github-enterprise-server.md
/tmp/gh-docs/content/get-started/start-your-journey/creating-an-account-on-github.md
```

`--json` emits the full hit list as pretty-printed JSON, including the
chunk text and score (text elided below for brevity):

```sh
mq search "markdown syntax" --json -n 2
```

```json
[
  {
    "chunk_index": 25,
    "id": "50d8611f967c51867f3f266a93e1a53d72bdf1ac9449fbfd0709273e40e98dae",
    "path": "/tmp/gh-docs/content/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax.md",
    "score": 7.739250659942627,
    "text": "`Let's rename \\*our-new-project\\* to \\*our-old-project\\*.` ...",
    "uri": "markq://default/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax.md"
  },
  {
    "chunk_index": 1,
    "id": "7b4aed60b808b3038cd6355bda8446d516486e0d5fdc02764e47efa42df249ec",
    "path": "/tmp/gh-docs/content/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions.md",
    "score": 7.469119071960449,
    "text": "Mathematical expressions rendering is available in ...",
    "uri": "markq://default/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions.md"
  }
]
```

`--min-score <F>` drops hits below the threshold (applied *before* the
top-k cap so the cap counts only surviving rows):

```sh
mq search "command palette" --min-score 8.0 -n 4
```

```
  1.  10.045  markq://default/accessibility/github-command-palette.md#0
     --- title: GitHub Command Palette intro: 'Use the command palette to navigate, search, and run comma…
  2.   9.768  markq://default/accessibility/github-command-palette.md#10
     ### Keystroke functions These keystrokes are available when the command palette is in navigation and…
  3.   9.734  markq://default/accessibility/github-command-palette.md#3
     The ability to run commands directly from your keyboard, without navigating through a series of menu…
  4.   9.599  markq://default/accessibility/github-command-palette.md#2
     When you open the command palette, the suggestions are optimized to give you easy access from anywhe…
```

`--all` removes the top-k cap entirely: instead of capping at `--top-k`
(`-n`), markq requests as many hits as there are rows in the dataset, so
every match above the score floor is returned rather than just the top `k`.
Useful for piping every match into another tool.

## 5. Hyphenated identifiers recall correctly

LanceDB's FTS tokenizer (tantivy) splits hyphenated terms on both the
index and query sides, so identifiers like `two-factor` and
`command-line` match without any query-side sanitizer:

```sh
mq search "two-factor" -n 3
```

```
  1.  10.893  markq://default/using-github/github-mobile.md#1
     * Notifications for {% data variables.product.prodname_mobile %}, see [AUTOTITLE](/account-and-profi…
  2.  10.880  markq://default/onboarding/getting-started-with-your-github-account.md#5
     {% endif %} {% ifversion ghes %} The administrator of your {% data variables.product.prodname_ghe_se…
  3.  10.726  markq://default/onboarding/getting-started-with-github-team.md#9
     You can help to make your organization more secure by recommending or requiring two-factor authentic…
```

```sh
mq search "command-line" -n 3
```

```
  1.   5.561  markq://default/learning-to-code/getting-started-with-git.md#11
     {% data variables.product.prodname_desktop %} is designed to address your day-to-day Git needs. As y…
  2.   5.328  markq://default/learning-to-code/getting-started-with-git.md#12
     1. In {% data variables.product.prodname_desktop %}, press <kbd>Ctrl</kbd>+<kbd>`</kbd> to open your…
  3.   5.273  markq://default/learning-to-code/getting-started-with-git.md#10
     You now have all of the skills necessary for setting up and using Git on a project! ## Diving deeper…
```

Because the tokenizer splits on non-alphanumerics symmetrically, no
query-side hyphen handling is needed: the indexed `two-factor` and the
query `two-factor` tokenize identically.

The hyphen is therefore just a delimiter, like a space — searching
`command-line` and `command line` both reduce to the two terms
`command` and `line`, so they return the **exact same hits with the
exact same scores**:

```sh
mq search "command line" -n 3   # identical ranking and scores to "command-line" above
```

```
  1.   5.561  markq://default/learning-to-code/getting-started-with-git.md#11
     {% data variables.product.prodname_desktop %} is designed to address your day-to-day Git needs. As y…
  2.   5.328  markq://default/learning-to-code/getting-started-with-git.md#12
     1. In {% data variables.product.prodname_desktop %}, press <kbd>Ctrl</kbd>+<kbd>`</kbd> to open your…
  3.   5.273  markq://default/learning-to-code/getting-started-with-git.md#10
     You now have all of the skills necessary for setting up and using Git on a project! ## Diving deeper…
```

## 6. Embed the corpus — `markq embed`

`embed` walks rows where `embedding IS NULL`, fills them via the
Qwen3-Embedding-0.6B Q8_0 GGUF (downloaded once into
`~/.cache/markq/models/`), then rebuilds the HNSW vector index and the
BM25 FTS index so both stay consistent with the rewritten rows.

```sh
mq embed
```

```
embedded 740 row(s) over 3 batch(es) (model=Qwen/Qwen3-Embedding-0.6B-GGUF/Q8_0, dim=1024)
```

On the `--features vulkan` build (confirm with `markq --version`, see
[§Setup](#setup)) the model offloads to the GPU. With `RUST_LOG=info` the device
selection is visible (this run used an NVIDIA T1200 over Vulkan; the
default log level keeps it quiet):

```
INFO load_from_file: llama-cpp-2: using device Vulkan1 (NVIDIA T1200 Laptop GPU) (0000:01:00.0) - 3954 MiB free
```

`inspect` now shows the embedder metadata recorded on the dataset:

```sh
mq inspect | tail -3
```

```
lancedb_crate_version:     0.27.2
embedder_model:            Qwen/Qwen3-Embedding-0.6B-GGUF/Q8_0
embedder_dim:              1024
```

Re-running `embed` is a no-op (the scan finds nothing with
`embedding IS NULL`):

```sh
mq embed
```

```
embedded 0 row(s) over 0 batch(es) (model=Qwen/Qwen3-Embedding-0.6B-GGUF/Q8_0, dim=1024)
```

## 7. Vector search — `markq vsearch`

Cosine k-nearest neighbors (KNN) over the embedding column. The query
string is embedded with the same model recorded in
`markq.embedder_model`, then run against the `IvfHnswSq` index — an
inverted file (IVF) of HNSW graphs with scalar quantization (SQ) —
built during `embed`.

```sh
mq vsearch "how do I review someone else's code changes" -n 3
```

```
  1.   0.368  markq://default/learning-to-code/getting-feedback-on-your-code-from-github-copilot.md#0
     --- title: Getting feedback on your code from GitHub Copilot shortTitle: Getting feedback on your co…
  2.   0.235  markq://default/using-github/communicating-on-github.md#2
     * Are useful for discussing specific details of a project such as bug reports, planned improvements …
  3.   0.207  markq://default/using-github/github-flow.md#5
     Continue to make, commit, and push changes to your branch until you are ready to ask for feedback. >…
```

The score column is cosine similarity (`1 - distance`); higher = better,
range `[-1, 1]`. Note the score *scale* differs from BM25 — BM25 scores
are log-domain and unbounded; cosine sits in a tight bounded range.
That's exactly the kind of cross-list normalization mismatch that
the RRF fusion in `markq query` exists to paper over.

`--json` returns the full structured rows (text elided here):

```sh
mq vsearch "how do I review someone else's code changes" -n 2 --json
```

```json
[
  {
    "chunk_index": 0,
    "id": "e7f3cf136fa2e0eace291b08869d3b79b1a8feacb4f9e5e485b2695a97308d5e",
    "path": "/tmp/gh-docs/content/get-started/learning-to-code/getting-feedback-on-your-code-from-github-copilot.md",
    "score": 0.36768442392349243,
    "text": "--- title: Getting feedback on your code from GitHub Copilot ...",
    "uri": "markq://default/learning-to-code/getting-feedback-on-your-code-from-github-copilot.md"
  },
  {
    "chunk_index": 2,
    "id": "238b723b0051063f153d506ec105d00c6d23286d2e1653febb5627d1f0f12c61",
    "path": "/tmp/gh-docs/content/get-started/using-github/communicating-on-github.md",
    "score": 0.2353009581565857,
    "text": "* Are useful for discussing specific details of a project ...",
    "uri": "markq://default/using-github/communicating-on-github.md"
  }
]
```

`--files`, `--min-score`, `--all`, and `--top-k` (`-n`) work identically to
`markq search` (the formatter module is shared). `--explain` and
`--collection` (`-c`) are recognized but gated until that work lands.

`vsearch` against a dataset that hasn't been embedded errors clearly
without loading the model:

```sh
mq --dataset /tmp/unembedded.lance vsearch "anything"
```

```
Error: no embeddings in this dataset; run `markq embed` first to populate them
```

## 8. Hybrid retrieval — `markq query`

`markq query` runs BM25 and vector KNN concurrently and fuses them with
weighted RRF:

```sh
mq query "how do I review someone else's code changes" -n 5
```

```
  1.   0.092  markq://default/learning-to-code/getting-feedback-on-your-code-from-github-copilot.md#0
     --- title: Getting feedback on your code from GitHub Copilot shortTitle: Getting feedback on your co…
  2.   0.062  markq://default/onboarding/getting-started-with-your-github-account.md#18
     A fork is a copy of a repository that you manage, where any changes you make will not affect the ori…
  3.   0.032  markq://default/learning-to-code/developing-your-project-locally.md#5
     If the README doesn't include information about installing dependencies, you can: * **Look for confi…
  4.   0.030  markq://default/learning-to-code/reusing-other-peoples-code-in-your-projects.md#12
     With this tutorial, you've learned how to safely reuse other people's code in your own work. To cele…
  5.   0.030  markq://default/using-github/communicating-on-github.md#2
     * Are useful for discussing specific details of a project such as bug reports, planned improvements …
```

`--explain` adds a per-stage timing summary and a contribution trace on
stderr (so stdout stays pipe-clean):

```sh
mq query "how do I review someone else's code changes" --explain -n 5 2> trace.txt
```

```
bm25:   20 hits in 7ms
embed:  query in 65ms
vector: 20 hits in 5ms
fuse:   37 unique docs in 0ms

rank  id               final      lex(rank,w)      vec(rank,w)   bonus
1     e7f3cf136fa2    0.0919       ( 2, 0.75)       ( 1, 0.60)    0.07
2     e47612ab4d1b    0.0623       ( 1, 0.75)     (  - ,   - )    0.05
3     c9fe168f4c4c    0.0319       ( 3, 0.75)     (  - ,   - )    0.02
4     7d217210d0fc    0.0297     (  - ,   - )       ( 2, 0.60)    0.02
5     238b723b0051    0.0295     (  - ,   - )       ( 3, 0.60)    0.02
```

The trace shows fusion at work: the top doc won the vector list and
placed #2 in BM25, and that cross-list agreement plus the top-rank bonus
lift it above the doc that won BM25 outright (`e47612ab4d1b`, which never
appears in the vector list). See
[`markq.md`](markq.md) for the full RRF scoring details.

## 9. Cross-encoder reranking — `markq rerank` and `query --rerank`

The stages above rank by lexical or vector similarity (and their RRF
fusion). A cross-encoder reranker adds a precision pass: it reads each
`(query, chunk)` pair *together* through one model and scores how well
the chunk answers the query. markq uses Qwen3-Reranker-0.6B Q8_0
(downloaded once into `~/.cache/markq/models/`, offloaded to the GPU on a
`vulkan` build just like the embedder). The score is `P("yes")` in
`[0, 1]` — the model's confidence, as a yes/no judge, that the chunk is
relevant.

### Integrated — `query --rerank`

`--rerank` reranks the fused hits *after* fusion, in one command. Compare
the hybrid (RRF) order with the reranked order for the same query:

```sh
mq query "how do I undo the last commit" -n 5            # hybrid (RRF) order
```

```
  1.   0.071  markq://default/using-git/using-git-rebase-on-the-command-line.md#0
     --- title: Using Git rebase on the command line redirect_from: - /articles/using-git-rebase - /artic…
  2.   0.060  markq://default/learning-to-code/getting-started-with-git.md#13
     When {% data variables.product.prodname_copilot_short %} asks what kind of command you're looking fo…
  3.   0.041  markq://default/using-git/about-git-rebase.md#2
     To rebase all the commits between another branch and the current branch state, you can enter the fol…
  4.   0.040  markq://default/using-git/about-git-rebase.md#3
     To rebase the last few commits in your current branch, you can enter the following command in your s…
  5.   0.040  markq://default/using-git/using-git-rebase-on-the-command-line.md#1
     Git gets to the `edit dd1475d` operation, stops, and prints the following message to the terminal: `…
```

```sh
mq query "how do I undo the last commit" --rerank -n 5   # reranked order
```

```
  1.   0.805  markq://default/using-git/using-git-rebase-on-the-command-line.md#1
     Git gets to the `edit dd1475d` operation, stops, and prints the following message to the terminal: `…
  2.   0.722  markq://default/using-git/about-git-rebase.md#3
     To rebase the last few commits in your current branch, you can enter the following command in your s…
  3.   0.239  markq://default/using-git/resolving-merge-conflicts-after-a-git-rebase.md#0
     --- title: Resolving merge conflicts after a Git rebase intro: 'When you perform a `git rebase` oper…
  4.   0.234  markq://default/using-git/about-git-rebase.md#4
     <dt><code>fixup</code></dt> <dd>This is similar to <code>squash</code>, but the commit to be merged …
  5.   0.206  markq://default/using-git/about-git-rebase.md#0
     --- title: About Git rebase redirect_from: - /rebase - /articles/interactive-rebase - /articles/abou…
```

RRF put the page's title/frontmatter chunk (`…on-the-command-line.md#0`)
first because it's lexically dense. The cross-encoder demotes it and
lifts the chunk that actually explains stopping at an `edit` operation to
rewrite a commit (`#1`) to the top, keeps the concrete "rebase the last
few commits" instructions, and pulls in
`resolving-merge-conflicts-after-a-git-rebase.md` — which wasn't in the
hybrid top-5 at all. Note the score scale changes: the leading column is
no longer the RRF score (≈0.04–0.07) but the reranker's `P("yes")`
(≈0.2–0.8).

Fusion itself is unchanged — `--rerank` only reorders what fusion
produced. To keep it a fast, small-set precision stage, `query --rerank`
reranks at most the top 64 fused candidates and bounds retrieval depth to
match, so `--rerank --all` stays cheap.

### Standalone — pipe any retrieval output through `markq rerank`

`markq rerank` reads a JSON candidate array on stdin — the `--json`
output of `search`, `vsearch`, or `query` — so retrieval and reranking
compose as a Unix pipe. The query is passed explicitly with `--query` (it
is not inferred from the payload), and the result cap is spelled `--top-k`
— unlike `search`/`vsearch`/`query`, the `rerank` subcommand defines no
`-n` short alias, so `--top-k` is the only spelling here:

```sh
mq query "how do I undo the last commit" --json -n 5 \
  | mq rerank --query "how do I undo the last commit" --json --top-k 3
```

```json
[
  {
    "id": "1afdc769c27e49c9b0f8471d8946d8cf5880a40c231af05ede0561b44d06a715",
    "text": "Git gets to the `edit dd1475d` operation, stops, and pr ...",
    "rerank_score": 0.80497605,
    "rank": 1,
    "chunk_index": 1,
    "path": "/tmp/gh-docs/content/get-started/using-git/using-git-rebase-on-the-command-line.md",
    "score": 0.039691708981990814,
    "uri": "markq://default/using-git/using-git-rebase-on-the-command-line.md"
  },
  {
    "id": "8bb340f869974de559df442114ec44e60e045a7589a89b12a595ce10ff067157",
    "text": "To rebase the last few commits in your current branch,  ...",
    "rerank_score": 0.7220729,
    "rank": 2,
    "chunk_index": 3,
    "path": "/tmp/gh-docs/content/get-started/using-git/about-git-rebase.md",
    "score": 0.04008718952536583,
    "uri": "markq://default/using-git/about-git-rebase.md"
  }
]
```

Each result keeps its `id` and gains a `rerank_score` and a 1-based
`rank`; the first-stage fields (`score`, `path`, `uri`, `chunk_index`)
pass through untouched, so a reranked list can itself be piped onward.
`rerank` reads stdin and never opens the dataset, so the `--dataset` the
`mq` helper adds is a harmless no-op.

One difference from the integrated form: standalone `rerank` scores
exactly the candidates it is handed (here, the query's top 5), whereas
`query --rerank` reranks the full fused pool before top-k — which is why
the integrated run above could surface a chunk the piped top-5 never
included.

### Edge cases

An empty candidate array is a no-op — empty output, exit 0, and the model
is never loaded:

```sh
echo '[]' | mq rerank --query "anything"    # (no output, exit 0)
```

A single candidate is returned as-is (still scored):

```sh
echo '[{"id":"only","text":"Git rebase lets you rewrite commit history."}]' \
  | mq rerank --query "how do I undo a commit"
```

```
  1.   0.090  only
     Git rebase lets you rewrite commit history.
```

## 10. What's still gated — the final surface is visible

These commands all parse but exit with a structured message, so a demo
viewer can see exactly which slice ships next:

```sh
mq search "x" --explain
# Error: --explain is not implemented yet

mq search "x" -c notes
# Error: collection filtering is not implemented yet; omit -c for now
```

`models`, `doctor`, `compact`, `get`, `multi-get`, `collection`,
`context`, `serve`, `status`, `config`, and `watch` all behave the same
way — registered in the clap surface, gated at the call site until their
slice lands. `query`, `query --rerank`, and standalone `rerank` all work;
the `--explain` / `--collection` (`-c`) flags on `search` and `vsearch` are
still gated.
