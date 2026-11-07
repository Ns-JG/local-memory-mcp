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
uv run pytest                        # 303 tests, ~3s, offline
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

The operator CLI (`uv run local-memory <command>`) and the transports:

```sh
uv run local-memory doctor                        # store/model consistency check
uv run local-memory config [--json]               # resolved configuration
uv run local-memory export-config <path>          # copy your config.toml to <path>
uv run local-memory stats                         # record counts, model, kinds
uv run local-memory search "query" [-k N] [--scope S] [--kind K] [--detail titles|snippets|full]
uv run local-memory get <id>                      # one record in full
uv run local-memory browse [--scope S] [--kind K] [--status ST] [--search Q] [-n N] [--no-interactive]
uv run local-memory reindex [--model M] [--all]   # required after changing model
uv run local-memory models                        # embedding model registry
uv run local-memory kinds                         # seeded vs in-use record kinds
uv run local-memory scopes [path]                 # scope mappings, or resolve a path
uv run local-memory purge [--yes]                 # drop tombstones past retention
local-memory-stdio                                # MCP server over stdio (what agents spawn)
local-memory-http                                 # MCP server over loopback HTTP
```

Smoke-test the viewer without a terminal (`TERM=dumb` or `CI=1` forces static
mode):

```sh
CI=1 uv run local-memory browse --scope work
```

## Layout and the one layering rule

```
config.py     Config (pydantic) + ConfigManager: reads and validates config.toml
models.py     MemoryRecord, SearchHit, Status. errors.py, utils.py: leaf modules
embedders/    fastembed wrapper, model registry, query/document prefixes
storage/      schema.py (tables, migrations)  db.py (connection, retry)
              lexical.py (FTS5)  store.py (record CRUD, kinds, vectors)
retrieval/    dense.py (cosine scan)  fusion.py (RRF)
service.py    search / get / store / forget / confirm   <- the product
app.py        composition root: config -> store -> embedder -> service
browse/       read-only terminal viewer over Store       <- reads only
mcp/          server.py (tools, thin adapters)  stdio.py / http.py  transports
cli.py        operator CLI
```

Arrows point one way: `config`/`errors`/`models` <- `storage`, `embedders` <-
`retrieval` <- `service` <- `mcp`, `cli`, `browse`. Only `service` and `app` know
about config; `storage`, `retrieval` and `embedders` get plain values.
`MemoryService` takes a config *provider* (`ConfigManager.load`), called per
operation, so edits to `config.toml` (new scopes, thresholds) apply without a restart.

**All retrieval and mutation logic lives in `service.py`.** `mcp/server.py` and
`cli.py` unpack arguments and format output; a tool that embeds, filters or ranks
is a bug, because it forks behaviour that stdio and HTTP must share.

`browse/` is the deliberate exception: it reads `Store` directly rather than
`service`, because listing records must not pay for constructing a `DenseIndex`
and an ONNX session. It calls `service` for search only. Do not "fix" this by
routing listings through the service.

`browse/terminal.py` is the only module allowed to import `termios`/`tty`.

## Things that break silently

**stdout is the protocol channel.** For stdio, the client spawns the process and
parses stdout as MCP frames. A stray `print()` corrupts the stream and the client
reports an opaque parse error. Diagnostics go to stderr. This is why
`mcp/stdio.py`'s `main()` catches everything and returns 1 rather than raising into the
client.

**Schema changes need a migration.** The schema the code expects is the tracked
`SCHEMA_VERSION` file at the repo root; each database records its own version in
`PRAGMA user_version`. `storage/schema.py:migrate_to_schema` creates a fresh
database at the target and otherwise runs `MIGRATIONS[n]` (n -> n+1) in order, each
in a transaction. A missing handler fails with `no migration path from schema version
N to N+1`, and a database newer than the code is refused. Editing `SCHEMA` without
bumping `SCHEMA_VERSION` and adding a handler leaves existing databases on the old
schema: `CREATE TABLE IF NOT EXISTS` will not alter an existing table.

**`browse` must never write.** The read-only guarantee is enforced by a test
(`_WriteWatch` in `tests/test_browse.py`) that installs a SQLite trace callback
and fails on any statement matching `_WRITE`. It uses `set_trace_callback`
because `Connection.execute` is a read-only attribute and cannot be
monkeypatched; the callback also sees statements from `executescript` and from
triggers. `_WRITE` has its own self-tests — if you widen the regex, widen those
too. Adding a write to `browse/` means breaking that test, which is the point.

**The model cache directory comes from `config.model_cache`** and is passed to
`build_embedder`, which hands it to fastembed explicitly. fastembed's own default is
a temp directory macOS reaps, which silently re-downloads the model every few weeks.
The `fastembed` import is also deferred inside functions so tests can inject a fake
embedder without touching the network - keep it that way.

**Query prefixes go on queries only, document prefixes on documents only.**
`embed_query` applies `ModelSpec.query_prefix` and `embed_documents` applies
`document_prefix`. Mixing them up (or applying both to a query) degrades retrieval
with no error. See GLOSSARY.

**CAS uses `rev`, never `updated_at`.** `updated_at` has one-second resolution,
so two writes in the same second are indistinguishable and the swap would pass
when it must fail. A silently-passing CAS is worse than none.

**Do not open the database `mode=ro` or `immutable=1`.** It is WAL, so a
read-only connection cannot create the `-shm` file it needs and fails with
`unable to open database file (14)`; `immutable=1` opens but ignores committed
WAL transactions and shows stale data. `browse` opens read-write and issues only
`SELECT`s. Details in GLOSSARY.

**The HTTP transport has no authentication**, which is why `mcp/http.py`'s `main()`
refuses any non-loopback bind and sets `host_origin_protection=True` explicitly
(FastMCP 4.0.10 defaults it to `False`). Do not add a bind-all flag, and do not
drop the origin check. The host and port come from `[server]` in `config.toml`.

## Testing conventions

- **Never let a test download a model.** `tests/conftest.py` provides
  `FakeEmbedder` (256-dim hashed unigrams **plus bigrams**) and the `store`,
  `service`, `embedder`, `config` and `config_manager` fixtures. Bigrams are load-bearing: unigram-only
  vectors at 256 dims pushed unrelated sentences over the 0.92 dedup threshold
  and collapsed distinct records. If you add a test that needs vectors, use the
  fixture and match `FAKE_DIM`.
- `asyncio_mode = "auto"`, so async tests need no marker or explicit fixture
  loop scope.
- HTTP tests bind a real socket on a free port and skip if the sandbox forbids
  binding. Do not mock the transport instead.
- **The browse paint path cannot be unit-tested**: `rich.Live` does not repaint
  to a non-terminal, so a `BytesIO` harness passes while the terminal is broken.
  Tests inject `browse.terminal._RecordingRenderer`; the real paint path was verified by
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
- Scopes and kinds are **config-driven**: `[scopes]` in `config.toml` lists
  `available` scopes, a `default`, and a `[scopes.<name>]` table of paths for
  each. `config.toml` at the repo root is only the template; the user's copy lives
  in `~/.config/local-memory/config.toml` and is re-read on every call. The agent-facing MCP
  instructions are `local_memory_instructions.md` beside it (`instructions_path`, relative to the config file);
  `scripts/install_config.sh` installs both (`--uninstall` backs the config up to the
  gitignored `backups/`); `local-memory export-config <path>` copies the active config out. A record
  whose scope is later removed from the config stays valid; only writes validate.
- `seed_kinds` in the config is a seed list, not an enum. `local-memory kinds` counts from
  live records because the `kinds` table remembers every kind ever written.
- A missing scope path is a warning, not an error — a moved project directory
  should be reported, not fatal.