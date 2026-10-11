# Glossary

Terms that are easy to forget, or that look wrong until you remember what they
mean. Everything here is either something this project uses deliberately or
something that will confuse you if you meet it in a schema dump.

For why a choice was made, see [DESIGN.md](DESIGN.md).

---

## Storage

### WAL (Write-Ahead Logging)

Journal mode where writes are appended to a `-wal` sidecar file and the main
database file is updated later, at a checkpoint.

The reason it is on (`store.py`, `PRAGMA journal_mode=WAL`): **readers do not
block writers, and writers do not block readers.** Two agents on one database —
OpenCode and Claude Code at the same time — is the whole reason. In the default
rollback journal, a reader holding a shared lock blocks the next write for as
long as it lives, which turns a long-running browse into a stall for the agent.

The cost: readers must consult the `-wal` file, which needs shared memory to do
efficiently. That is the `-shm` file, and it is the source of the read-only trap
below.

Checkpointing moves `-wal` contents into the main file. `local-memory` does not
checkpoint on a schedule; SQLite checkpoints automatically, and `purge` is the
only place it does so explicitly.

### `-wal`, `-shm`, `-journal`

Three sidecars next to `memory.db`:

| File | Purpose | Safe to delete? |
| --- | --- | --- |
| `memory.db-wal` | committed transactions not yet folded into the main file | **No** — deleting loses committed data |
| `memory.db-shm` | shared-memory index over the WAL | **No**, unless no process has the DB open |
| `memory.db-journal` | rollback journal; only in non-WAL mode | Only while nothing has it open |

They disappear when the last connection closes cleanly. If you see a stray
`-wal` with a non-zero size and nothing running, the previous process died
without closing; SQLite recovers it automatically on next open.

### `mode=ro` does not work on a WAL database

The single most confusing thing about WAL:

```
$ sqlite3 "file:$HOME/.local/share/local-memory/memory.db?mode=ro" "select 1;"
Parse error: unable to open database file (14)
```

A read-only connection **cannot create the `-shm` file**, and WAL requires it.
It works only if a writable-capable process already has the database open, so
the file exists.

`immutable=1` does open, and is a trap: it tells SQLite the file cannot change,
so it **skips the WAL entirely**. You get a silently stale view of the data. Do
not use it on a live database.

If you need a read-only handle, either open normally and promise yourself not to
write, or snapshot the file first. `local-memory browse` does the former: it
opens read-write and issues only `SELECT`s, which a test enforces.

### `busy_timeout`

How long a blocked statement waits for a lock before raising
`SQLITE_BUSY` instead of failing instantly (`store.py`, 5000 ms).

WAL allows only one writer at a time. With two agents writing, one will
momentarily block. This setting is what turns that from an error into a short
wait. It is paired with full-jitter retry in `Store._write`, because a busy
timeout alone still produces a retry stampede.

### Checkpoint

Folding `-wal` contents back into the main database file. `PRAGMA
wal_checkpoint(TRUNCATE)` also empties the `-wal` file.

This is a **write**. Any tool that runs it is a writer, even if it only meant to
look.

### `synchronous`

How strictly SQLite insists on durability at each commit (`NORMAL` here).

`FULL` fsyncs on every commit and is slow. `NORMAL` in WAL mode fsyncs only at
checkpoints, so a power loss can lose the last transactions without corrupting
the file. For a memory store that is rebuildable from conversation history, that
is the right trade.

### Page and `page_size`

SQLite's I/O unit, 4096 bytes here. Pages are the unit of allocation and of the
WAL. Relevant mostly when reading `dbstat` output to find what is using space.

---

## Schema objects

### Rowid

SQLite's implicit integer primary key. Every table has one unless declared
`WITHOUT ROWID`.

`records.rowid` is not the same as `records.id`. `records` declares
`rowid INTEGER PRIMARY KEY`, which makes the two names for one column — but
that column is the *rowid*, an internal handle, whereas `id` is a UUIDv7 that is
stable and survives export to another device. `records_fts` joins on **rowid**,
because that is what FTS5 indexes.

### `WITHOUT ROWID`

Declares a table whose primary key *is* the row identity, removing the separate
rowid table. Saves space and an indirection when the table is naturally keyed —
`records_fts_idx` and `records_fts_config` use it, because a lookup by key is
the only thing either ever does.

Note that a `TEXT PRIMARY KEY` is *not* the same thing: `kinds` and `meta` are
ordinary rowid tables that happen to have an automatic index (`sqlite_autoindex_*`)
enforcing uniqueness. They keep a rowid nobody reads.

### AUTOINCREMENT

Tells SQLite to never reuse a rowid, even after deletion, by recording the
highest ever used. Not used here: nothing depends on monotonic rowids surviving
deletion.

### External-content FTS5 table

```sql
CREATE VIRTUAL TABLE records_fts USING fts5(
  title, gist, body, tags,
  content='records', content_rowid='rowid',
  tokenize="unicode61 remove_diacritics 2 tokenchars '_'"
);
```

A virtual table (below) that **stores no text of its own**. `content='records'`
says the authoritative text lives in `records`, and `content_rowid='rowid'`
says how to find a row there.

The payoff: the body text is stored once, not twice. The cost is a correctness
obligation — the index can silently disagree with the table unless every insert,
update and delete is mirrored into the FTS table, which is what the three
triggers do. A stale lexical index returns records that no longer exist, and no
error is raised.

### Shadow tables

The ordinary tables SQLite creates behind the scenes to implement a virtual
table. `records_fts` owns five of them, and they are **not yours to query or
modify**:

| Table | Contents |
| --- | --- |
| `records_fts_data` | the inverted index, as blobs |
| `records_fts_idx` | term → posting list segments |
| `records_fts_docsize` | per-document token counts, for length normalisation |
| `records_fts_config` | tokenizer and rank configuration |
| `records_fts_content` | *only if* `content=` is not set — absent here |

If you see them in a schema dump, that is normal. Never `UPDATE` one. To rebuild
the index from scratch, drop and recreate the virtual table, or run the
`'rebuild'` command:

```sql
INSERT INTO records_fts(records_fts) VALUES('rebuild');
```

### Virtual table

A table implemented by a loadable extension rather than by the b-tree engine.
`records_fts` is one; `fts5` is the extension. Virtual tables appear in
`sqlite_master` with `type='table'` and show up in `\d`-style output like normal
ones, but ordinary SQL rules (constraints, foreign keys) do not apply to them.

### Trigger

Code SQLite runs automatically on a table event. This schema has three, all
feeding FTS5:

| Trigger | On | Does |
| --- | --- | --- |
| `records_ai` | INSERT | copy the new row into `records_fts` |
| `records_au` | UPDATE | delete the old row from FTS, insert the new |
| `records_ad` | DELETE | delete the row from FTS |

`AFTER` means the base-table change has already happened. The FTS delete uses
the `'delete'` command with the **old** column values, because FTS5 must remove
exactly the terms it indexed — removing by rowid alone would not remove the old
postings.

### `tokenize`

How FTS5 turns text into terms. Here:

- `unicode61` — Unicode-aware, case-folded.
- `remove_diacritics 2` — folds diacritics, so `Wroclawski` matches
  `Wrocławski`. Relevant given Polish content.
- `tokenchars '_'` — keeps `_` inside words, so `pg_bouncer` and `uv_sync` stay
  single terms.

### `bm25()`

FTS5's relevance function. Returns a **negative** number where *more negative is
better*, which is the opposite of the usual convention. `Store._fts_hits`
negates it so higher means better, consistently with the cosine channel.

Arguments are per-column weights, applied positionally in declaration order:
`(title 10, gist 5, body 1, tags 3)` — `BM25_WEIGHTS` in `store.py`. A title is a
record's compressed identity, so it dominates; tags outrank the body because a
body is where most words are least identifying.

### `MATCH` and the query language

FTS5's `MATCH` is not SQL. `AND`/`OR`/`NOT`/`NEAR` are operators, and bare words
are implicitly `AND`. Unquoted punctuation can be a syntax error — `foo-bar`
behaves surprisingly.

`Store._fts_tokens` extracts tokens with `[\w*]+` and quotes each one, which makes
FTS5 treat it as a literal and turns the sequence into an implicit `AND`. A
user's query containing an apostrophe or a `-` therefore cannot break search.
Note `*` is deliberately inside the token class, so FTS5 prefix search survives
(`pgb*`).

### Implicit AND, and the OR fallback

Strict AND is high precision but brittle: `postgresql connection` matches
**nothing** in a record that says `postgres`. `Store.lexical_search` runs AND
first and, if it under-fills, tops up with an OR pass, keeping AND results
first. Without the fallback the lexical channel silently disappears for ordinary
questions.

### Implicit indexes

`sqlite_autoindex_*` are created automatically to enforce `PRIMARY KEY` and
`UNIQUE` on non-integer keys. You cannot drop them, hence the odd names in
`sqlite_master`.

---

## Storage details specific to this project

### BLOB and float32

`vectors.vec` is a raw `float32` array, little-endian, as a BLOB — 384 floats =
1536 bytes for the default model. There is no header or dimension field, so the
dimension must be known from elsewhere (`embedding_dim`, and the model
registry).

`numpy.frombuffer(vec, dtype=np.float32)` is how it is read back.

### Vectors are derived state

They can be rebuilt from `title + body` with `local-memory reindex`. They are
*not* the source of truth, and treating them as such is how a store ends up
holding text and embeddings from different models. Changing model requires
reindexing; mixing dimensions raises an explicit error rather than corrupting
search.

### `foreign_keys=ON`

SQLite does **not** enforce foreign keys unless this pragma is set per
connection. `vectors.id REFERENCES records(id)` is only a real constraint
because the pragma is on. Without it, an orphaned vector is possible and
silently shifts every dense score.

### `store_epoch`

A row in `meta`, bumped on every commit. Readers compare it to the value they
saw when they loaded their in-memory index; a difference means reload. This is
how the HTTP server notices the stdio server wrote something — two processes,
one cache. See [DESIGN.md](DESIGN.md#cross-process-staleness).

### `rev`

Per-row version counter, incremented on every write, used as the
compare-and-swap token: `UPDATE ... WHERE id = ? AND rev = ?`, and a rowcount
of zero means someone else got there first.

It is deliberately *not* `updated_at`, which has one-second resolution — two
writes in the same second would carry identical values and the CAS would pass
when it had to fail. A CAS that silently passes is worse than none.

### Tombstone

A deleted record kept as a row with `status='deleted'` and `deleted_at` set.
Tombstones are excluded from search, retained for 730 days, and exist so that a
deletion can be propagated to other devices by a future sync. `purge` removes
expired ones permanently.

---

## Retrieval vocabulary

### Embedding / dense channel

A float vector per record representing meaning. Search compares by cosine
similarity. Recalls paraphrases; misses exact rare strings.

### Lexical / BM25 channel

Keyword matching with rarity weighting. Catches identifiers, error codes, file
names — things embeddings are notoriously bad at. Misses paraphrases.

### RRF (Reciprocal Rank Fusion)

Combines ranked lists without needing comparable scores:

```
score(d) = Σ_channels 1 / (k + rank_channel(d)),   k = 60
```

It consumes only *rank*, never raw score — which is exactly why it works here, as
a cosine and a BM25 score share no scale and no sensible normalisation. The
consequence is that a fused score is **not comparable across queries**, which is
why results report `dense_rank` and `lexical_rank` instead.

`k=60` is the value from Cormack et al. Larger values flatten the contribution of
top ranks.

### Progressive disclosure

Returning short previews from search and the full body only on request. Keeps
agent context from being consumed by records that turn out not to matter.

### Cosine similarity

Angle between vectors, ignoring magnitude. After L2-normalisation on write,
cosine is a dot product, so a full scan is one `matmul` and needs no per-query
normalisation.

Cosine always has a nearest neighbour, so **a nonsense query still returns the
closest records** rather than an empty list. Judging a hit therefore requires
looking at the ranks, not at the presence of a result. This is deliberate — a
similarity floor tuned on one corpus is a liability on another.

### Dedup threshold

0.92 by default. A new record at or above this cosine against a live record is
treated as a re-observation: the existing record's confidence rises and its tags
merge, instead of a near-identical row being created that competes in every
future search. High on purpose — a false merge loses information permanently.

---

## Project terms

### Scope

A label for where a record came from, derived by matching the project directory
against `[scope.*] paths` in `config.toml` (`work`, `uni`, `personal` in the
default config). Scopes are configuration, not a closed vocabulary: any key with
a `paths` list becomes one.

Filters run *before* retrieval, because a scope filter is cheaper than cosine
similarity and more precise for structured questions. A record stored with a
scope the current config does not define is still listed — only
`--scope` filtering and path inference depend on the mapping.

### Kind

An open vocabulary for record type (`pattern`, `decision`, `fact`, …), seeded but
agent-extensible. The `kinds` table remembers every kind ever written, so it is
not the same as "currently in use" — `local-memory kinds` counts from live
records.

### Supersede

Replacing a record with a newer one. The old row stays with
`status='superseded'` and `supersedes_id` pointing at the new one, so history
survives. Both rows are written in one transaction, so a half-applied
replacement cannot exist. Explicit supersession bypasses dedup: if you say this
replaces that, it does.

### FastEmbed / ONNX

`fastembed` runs embedding models through ONNX Runtime instead of PyTorch.
Consequence for this project: the first `embed_query` pays ~0.2s to initialise
the session. `configure_model_cache` sets `FASTEMBED_CACHE_PATH` before any model
is constructed, because fastembed reads that variable at construction time — set
it later and the model lands in a temp directory macOS reaps.

### Query prefix

BGE and E5 families expect an instruction on the **query** side only. E5's
registry entry also declares a `document_prefix`, which is *not* applied —
`embed_documents` sends text unmodified. For the BGE model in the default
registry that is correct, because it wants no prefix on documents. If a registry
entry is ever added whose model *does* require a document-side prefix, applying
it wrongly (or omitting it) degrades retrieval **silently**: no exception, just
worse results.

### UUIDv7

Time-ordered UUIDs. Sortable by creation, so `id DESC` approximates
newest-first, while remaining globally unique across devices.