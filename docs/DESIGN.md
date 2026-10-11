# Design

Why this is built the way it is, including the parts that are compromises.

## Constraints

Three shaped nearly every decision:

1. **Local-first is the point.** No query-time network calls, no API keys, no
   telemetry. The database is one file.
2. **Agent context is the scarce resource.** Search must not dump whole records.
3. **One person, two clients.** OpenCode and Claude Code, sometimes running at
   the same time, over one database.

## Pipeline

```
memory_search(query, scope?, kind?, k, detail)
  │
  ├─ candidate_ids(scope, kind, status=active)      ← filter first
  │
  ├─ embed query           → 384d, L2-normalised
  ├─ dense top-N           → brute-force cosine over an in-RAM matrix
  ├─ FTS5 BM25 top-N       → filtered by the same scope/kind
  │
  ├─ RRF fuse (k=60)       → score = Σ 1/(60 + rank)
  │
  └─ hydrate → previews, or full bodies on request
```

Filters run **before** retrieval, not after. They are cheaper than similarity
and more precise for structured questions like "what do you know about my
preferences".

## Storage

SQLite in WAL mode, one file, three tables of consequence:

| Table | Role |
| --- | --- |
| `records` | The data. Full text, metadata, status. |
| `records_fts` | FTS5 external-content index, kept in sync by triggers. |
| `vectors` | One float32 blob per record. Derived state, rebuildable. |
| `kinds` | Registry of kinds ever written, seeded or agent-coined. |
| `meta` | Schema version, and `store_epoch` (see below). |

### Why not a vector database

ChromaDB was the original implementation. It was dropped for this:

- **Vectors are derived.** They come from `title + body` through a model that
  can be swapped. Storing them in a system whose whole purpose is vector
  storage, separately from the text, means two things can disagree and no single
  transaction covers both.
- **Filters are the common case.** Scope and kind are cheap SQL predicates. Doing
  them in SQLite then masking the result set is simpler than teaching a vector
  engine about them.
- **Scale does not justify it.** The target is 10²–10⁴ records. A 15 MB float32
  matrix reloads in about 10 ms and a full scan is a single `matmul`.

`vectors.id` is a foreign key onto `records.id`, so an orphaned vector is a
database error rather than a silent inconsistency.

### FTS5 external content, kept honest

`records_fts` uses `content='records'` with `content_rowid='rowid'`, so the text
is stored once. Three triggers (`AFTER INSERT`/`UPDATE`/`DELETE`) keep it in
sync. This is a correctness obligation: a stale lexical index returns records that
no longer exist.

Column weights are `(title 10, gist 5, tags 3, body 1)`. A record's title is its
compressed identity, so it dominates.

### Query sanitisation

Every token is double-quoted before it reaches `MATCH`. Without that, a query
containing `"`, `*` or `NEAR` is a syntax error, and a query containing `'` is a
problem. Quoting makes FTS5 treat each token as a literal and turns the sequence
into an implicit AND.

That AND is brittle on its own: `postgresql connection` matches **nothing** in a
record that says `postgres`. So `lexical_search` runs AND first, and if it
under-fills, tops up with OR. AND results keep precedence. This was added after
watching ordinary questions silently lose their lexical channel.

## Retrieval

### Dense

Brute-force cosine over an in-RAM `float32` matrix. Vectors are L2-normalised on
write, so the scan is a plain `matmul` with no normalisation step at query time.

The matrix is a **cache of the `vectors` table**, never the source of truth.
Writes go durable-first, then to the cache.

### RRF

```
score(d) = Σ_channels 1 / (k + rank_channel(d)),  k = 60
```

`k=60` is the value from Cormack et al. RRF consumes only *rank*, never raw
score, which is the entire reason it works here: a cosine similarity and a BM25
score share no scale and no sensible normalisation.

The consequence is that fused scores are **not comparable across queries**, so
`SearchHit` returns the per-channel ranks (`dense_rank`, `lexical_rank`) instead.
Those are what let a caller judge a hit.

### Known behaviour: no similarity floor

Cosine always has a nearest neighbour, so a nonsense query still returns the
closest records rather than an empty list. This is intentional — a hard floor
tuned on one corpus is a liability on another — but it means callers must judge
results by rank, not by treating a hit as a match.

## Concurrency

Three layers, because two clients over one file needs all three.

1. **In-process.** A re-entrant lock serialises write transactions. Reads run
   lock-free under WAL.
2. **Cross-process.** `busy_timeout=5000` plus full-jitter retry on lock
   contention. Full jitter rather than equal jitter: with several waiters, equal
   jitter keeps them colliding in lockstep.
3. **Logical conflict.** Every update carries `WHERE rev = ?`. A rowcount of zero
   raises `ConflictError` rather than silently clobbering.

### `rev`, not `updated_at`

The first implementation compared `updated_at`. That was **wrong**: the timestamp
has one-second resolution, so two writes in the same second carry identical
values and the compare-and-swap passed when it had to fail. A CAS that silently
passes is worse than none, because it looks like protection.

`rev` is a per-row version counter, incremented on every write. `updated_at`
remains the wall clock, used only for last-writer-wins merges between devices.
`insert_and_supersede` writes both rows in one transaction precisely so a
half-applied replacement cannot exist.

### Cross-process staleness

Every commit bumps `meta.store_epoch`. A reader compares it to what it saw at
index-load time and reloads only when they differ. This is how the HTTP server
notices the stdio server wrote something.

## Deletion

```
active ──forget──> pending_delete ──approve──> deleted ──730d──> purged
                        │
                        └──dismiss──> active
```

`forget` removes the record from search immediately, so the requested effect is
visible, but nothing is destroyed until confirmed. `deleted` writes a tombstone
with `deleted_at`, which is what makes the deletion survive a future sync.

The MCP instructions explicitly tell the agent not to report a `memory_forget`
as a completed deletion.

## Duplicate collapse

Before inserting, the incoming record is embedded and compared against the live
index. A cosine at or above `dedup_threshold` (default 0.92) is treated as a
re-observation: the existing record's `confidence` rises, `times_confirmed`
increments, and tags merge. Otherwise a new row is created.

Without this, an agent that restates the same pattern five times leaves five rows
competing in every subsequent search. The threshold is high on purpose — a false
merge loses information permanently.

Explicit `supersedes_id` bypasses dedup: if you say this replaces that, it does.

## Embeddings

`fastembed`, ONNX, model cached to `~/.cache/local-memory/models`. The cache path
is set in `__init__` **before** fastembed is imported, because fastembed
otherwise caches into a temp directory that macOS reaps.

The query prefix is applied to queries only. BGE and E5 families expect an
instruction on the query side, and E5 additionally expects a different one on the
document side. Applying the query prefix to documents, or omitting it, degrades
retrieval **silently** — no exception, just worse results.

Every entry in the model registry is verified against
`TextEmbedding.list_supported_models()` by `tests/test_models.py`, name and dim
both. This exists because `BAAI/bge-m3` was once in the registry and does not
exist in fastembed: it resolved, appeared in `local-memory models`, and would
have failed only at model-load time.

Changing model requires `local-memory reindex`. Vectors from different models
cannot be compared, and the dimension mismatch is caught on load.

## Testing

260 tests, about six seconds, no network. The fake embedder is a hashed bag of
words with bigrams, not a stub returning constants — dedup and ranking are
cosine-similarity behaviours, so a constant-returning fake would let those tests
pass while the real code was broken. Bigrams specifically keep unrelated
sentences separable; unigrams alone at 256 dimensions collided above the dedup
threshold and collapsed distinct records into one.

The HTTP transport tests bind a real socket and skip if the sandbox forbids it.
The terminal viewer tests cannot do the same for the paint path — see
[The viewer](#the-viewer).

## Security

- No authentication anywhere. The HTTP transport therefore **refuses to bind
  anything but loopback**.
- `host_origin_protection=True` is set explicitly. FastMCP 4.0.10 defaults it to
  `False`, and binding to `127.0.0.1` does not stop a web page the user has open
  from POSTing to `127.0.0.1:8787/mcp`. Verified: a foreign `Origin` gets 403.
- FTS5 queries are parameterised and sanitised.
- No secrets in the record schema, so none can leak through an export.

## The viewer

`local-memory browse` is a **read path over `Store`**, not a database browser.
It reads through `Store.list_records` and `Store.get`, so it shows decoded tags,
the status lifecycle and confirmation counts — not 26 raw columns plus five
`records_fts_*` shadow tables and a float32 blob.

**It cannot write.** A viewer with write access would let an edit bypass `rev`,
`store_epoch` and vector recomputation, which is exactly how optimistic
concurrency silently stops working. Rather than relying on review, the
guarantee is asserted: `tests/test_browse.py` installs a SQLite trace callback
and fails if any statement matching `insert|update|delete|create|drop|pragma *=`
is executed across every read path, and separately checks that `store_epoch`
does not move. Tracing is used rather than monkeypatching `Connection.execute`
because that attribute is read-only — and the trace callback also sees
statements issued from inside `executescript` and triggers.

**It goes around the service, not through it.** `MemoryService.__init__`
constructs a `DenseIndex`, and the first `embed_query` initialises an ONNX
session costing ~0.2s. A listing has no business paying that, so `browse` uses
`Store` directly and defers the whole service behind `_LazySearcher`, which
builds it on first search and caches it. The test asserts the embedder is *not*
constructed by a listing.

**Search still runs through `MemoryService`**, deliberately: reimplementing the
dual-channel RRF pipeline for the viewer would be a second retrieval
implementation that drifts.

### Interactive terminal handling

This is the part that is easy to get subtly wrong, and the part a unit test
with a `BytesIO` cannot cover:

- **`sys.stdin` is text-mode.** `read(1)` returns `'j'`, not `b'j'`, so every
  byte-key lookup misses and every keystroke silently degrades to `"other"`.
  `browse._binary` resolves the `.buffer`. Regression-tested with a real
  `TextIOWrapper`, because a `BytesIO` fixture hides it completely.
- **`Live` renders exactly one renderable.** Passing a tuple raises
  `NotRenderableError` on a terminal — and only on a terminal; the test double
  stores tuples and never renders them. Frames are wrapped in `Group`.
- **The terminal needs cbreak.** Otherwise the tty buffers input a line at a
  time and echoes it into the middle of the table. `tty.setcbreak` clears
  `ICANON` and `ECHO` but leaves `ISIG`, so Ctrl-C still reaches the
  interpreter. Attributes are restored on any exit path, including a crash.
- **Text entry needs its own reader.** `read_key` maps bytes to actions, which
  is right for navigation and wrong for typing: a query containing `j` or `/`
  would be interpreted. `read_line` consumes raw characters instead and takes an
  `on_change` callback to repaint, otherwise the prompt blocks until submission
  and you type blind.

`_LiveRenderer` is therefore injectable, and `_RecordingRenderer` captures
frames in tests. The real paint path was verified separately by driving the
binary through a pty.

### Why not a web UI

A local web UI needs a port, which reopens the Host/Origin hardening question
already answered for the HTTP transport, and it hands the browser a copy of your
memory. A terminal UI has neither problem. The cost is text rendering, which for
a table of titles and short bodies is not a real loss.

## Known gaps

- **No sync yet.** Export/merge between devices is unimplemented, so the database
  is single-machine. Deliberately deferred until there is real data to exercise
  it against.
- **`constants.py` was removed** as a leftover from the ChromaDB implementation.
- **Dedup is threshold-based.** A paraphrase below 0.92 similarity still
  duplicates. Near-duplicate detection is a retrieval problem with a real
  precision/recall trade-off, not a threshold question.
- **One embedding per record**, over `title + body`. Tag text and gist are
  excluded: the gist is derived from the body, and tags are low-signal for
  similarity while already being weighted 3x in BM25.

## Layout

```
models.py     record schema, vocabularies, model registry, config
store.py      SQLite + FTS5 + vectors + optimistic writes
embedder.py   fastembed wrapper, query/document prefixes
index.py      dense matrix, cosine scan, RRF
service.py    search / get / store / forget / confirm   ← the product
browse.py     read-only terminal viewer over Store       ← reads only
server.py     MCP tool surface, thin adapters
cli.py        operator CLI
mcp_stdio.py  default transport
mcp_http.py   optional transport, loopback only
```

The service layer holds the logic. Transports only unpack arguments and format
output — no tool embeds, filters, or ranks. That is what keeps the stdio and HTTP
paths interchangeable and the logic testable in milliseconds.

`browse.py` is the deliberate exception: it reads `Store` directly rather than
`service`, because it has no business loading an embedding model to list
records. It calls `service` only for search.