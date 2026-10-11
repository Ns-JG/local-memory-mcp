# AGENTS.md

Local-first hybrid-retrieval memory service for coding agents, spoken over MCP.
Python 3.13, uv, SQLite + FTS5 + numpy. No network at runtime except the one-off
model download.

Read `docs/DESIGN.md` for *why* each choice was made and `docs/GLOSSARY.md` for
what the SQLite machinery (WAL, shadow tables, triggers) actually means. This
file only records what you would otherwise get wrong.

## Commands

```sh
uv sync                              # required once; creates .venv
uv run pytest                        # 260 tests, ~6s, offline
uv run ruff check .
uv run ruff format .

uv run pytest tests/test_store.py    # one file
uv run pytest -k lexical              # by name
uv run pytest tests/test_browse.py::test_read_key_accepts_a_text_wrapped_stream
```

`addopts = "-q"` is already set in `pyproject.toml`; pass `-v` if you want
output. There is **no typechecker, no coverage config, no pre-commit, no CI and
no Makefile/justfile** — do not go looking for one, and do not add a `mypy` step
to the verification loop unless asked.

Smoke-test the viewer without a terminal (`TERM=dumb` or `CI=1` forces static
mode):

```sh
CI=1 uv run local-memory browse --scope work
```

## Layout and the one layering rule

```
models.py     dataclasses, vocabularies, model registry, config loading. No I/O.
store.py      SQLite + FTS5 + vectors + optimistic writes
embedder.py   fastembed wrapper, query prefixes
index.py      dense matrix, cosine scan, RRF
service.py    search / get / store / forget / confirm   <- the product
browse.py     read-only terminal viewer over Store       <- reads only
server.py     MCP tools, thin adapters
cli.py        operator CLI        mcp_stdio.py / mcp_http.py  transports
```

**All retrieval and mutation logic lives in `service.py`.** `server.py` and
`cli.py` unpack arguments and format output; a tool that embeds, filters or ranks
is a bug, because it forks behaviour that stdio and HTTP must share.

`browse.py` is the deliberate exception: it reads `Store` directly rather than
`service`, because listing records must not pay for constructing a `DenseIndex`
and an ONNX session. It calls `service` for search only. Do not "fix" this by
routing listings through the service.

`browse.py` is also the only module allowed to import `termios`/`tty`.

## Things that break silently

**stdout is the protocol channel.** For stdio, the client spawns the process and
parses stdout as MCP frames. A stray `print()` corrupts the stream and the client
reports an opaque parse error. Diagnostics go to stderr. This is why
`mcp_stdio.main()` catches everything and returns 1 rather than raising into the
client.

**There is no migration path.** `Store._migrate` accepts `user_version == 0`
(create fresh) and rejects anything else with `no migration path from schema
version N to N+1`. So editing `SCHEMA` without bumping `SCHEMA_VERSION` leaves
existing databases on the old schema — `CREATE TABLE IF NOT EXISTS` will not
alter an existing table — and you get confusing runtime errors instead of a
clear failure. If you change the schema you must write an explicit migration.
The live DB currently holds 0 records, so deleting it is the cheap path.

**`browse` must never write.** The read-only guarantee is enforced by a test
(`_WriteWatch` in `tests/test_browse.py`) that installs a SQLite trace callback
and fails on any statement matching `_WRITE`. It uses `set_trace_callback`
because `Connection.execute` is a read-only attribute and cannot be
monkeypatched; the callback also sees statements from `executescript` and from
triggers. `_WRITE` has its own self-tests — if you widen the regex, widen those
too. Adding a write to `browse.py` means breaking that test, which is the point.

**`FASTEMBED_CACHE_PATH` is set before the model is constructed**, in
`embedder.configure_model_cache`. fastembed reads it at construction time; set it
later and the model lands in a temp directory macOS reaps, so it silently
re-downloads every few weeks. The `fastembed` import is also deferred inside
functions so tests can inject a fake embedder without touching the network —
keep it that way.

**Query prefixes go on queries only.** `embed_documents` sends text unmodified.
Applying the query prefix to documents degrades retrieval with no error. Note
`ModelSpec.document_prefix` exists in the registry but is **not applied** — that
is fine for the default BGE model and wrong for a model that wants one. See
GLOSSARY.

**CAS uses `rev`, never `updated_at`.** `updated_at` has one-second resolution,
so two writes in the same second are indistinguishable and the swap would pass
when it must fail. A silently-passing CAS is worse than none.

**Do not open the database `mode=ro` or `immutable=1`.** It is WAL, so a
read-only connection cannot create the `-shm` file it needs and fails with
`unable to open database file (14)`; `immutable=1` opens but ignores committed
WAL transactions and shows stale data. `browse` opens read-write and issues only
`SELECT`s. Details in GLOSSARY.

**The HTTP transport has no authentication**, which is why `mcp_http.main()`
refuses any non-loopback bind and sets `host_origin_protection=True` explicitly
(FastMCP 4.0.10 defaults it to `False`). Do not add a bind-all flag, and do not
drop the origin check. `LOCAL_MEMORY_HOST` / `LOCAL_MEMORY_PORT` override the
defaults.

## Testing conventions

- **Never let a test download a model.** `tests/conftest.py` provides
  `FakeEmbedder` (256-dim hashed unigrams **plus bigrams**) and the `store`,
  `service` and `embedder` fixtures. Bigrams are load-bearing: unigram-only
  vectors at 256 dims pushed unrelated sentences over the 0.92 dedup threshold
  and collapsed distinct records. If you add a test that needs vectors, use the
  fixture and match `FAKE_DIM`.
- `asyncio_mode = "auto"`, so async tests need no marker or explicit fixture
  loop scope.
- HTTP tests bind a real socket on a free port and skip if the sandbox forbids
  binding. Do not mock the transport instead.
- **The browse paint path cannot be unit-tested**: `rich.Live` does not repaint
  to a non-terminal, so a `BytesIO` harness passes while the terminal is broken.
  Tests inject `browse._RecordingRenderer`; the real paint path was verified by
  driving the binary through a pty. If you change rendering or key handling,
  re-verify in a pty — three separate production bugs (tuple passed to
  `Live.update`, missing `cbreak`, `sys.stdin` read as text instead of
  `sys.stdin.buffer`) passed every unit test and failed immediately on a real
  terminal.
- For text input specifically: a `BytesIO` fixture **cannot** catch the
  text-mode bug, because `b'j'` matches a byte-key table and `'j'` does not. Use
  `io.TextIOWrapper` when testing anything that reads `sys.stdin`.

## Repo state

- **The repository has no commits yet** (`main` has no `HEAD`; everything is
  untracked). Do not commit, amend, or otherwise touch git history unless
  explicitly asked.
- `.gitignore` covers `.venv` and `__pycache__` but **not** `.pytest_cache` or
  `.ruff_cache`, both of which exist in the working tree. Clean them up rather
  than letting them into a first commit.
- The scope vocabulary is **config-driven**, not a closed enum. `Scope` has four
  members, but the default `config.toml` only attaches paths to `work`, `uni`
  and `personal`; `general` is the catch-all `scope_for_path` falls back to when
  no path rule matches. So a record can legitimately have scope `general` even
  though nothing maps to it. Do not "tidy" the enum down to the configured
  three.
- `SEED_KINDS` is a seed list, not an enum. `local-memory kinds` counts from
  live records because the `kinds` table remembers every kind ever written.
- A missing scope path is a warning, not an error — a moved project directory
  should be reported, not fatal.