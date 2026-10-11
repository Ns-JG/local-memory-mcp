# local-memory

Local-first hybrid-retrieval long-term memory for coding agents, exposed over
MCP. Nothing leaves the machine: no API keys, no network calls at query time, no
telemetry. The database is a single SQLite file you can back up, inspect with
`sqlite3`, and delete.

## Why

An agent that forgets your preferences every session re-asks the same questions.
Most "agent memory" products solve this by sending your codebase and
conversations to a third-party API. This does it locally instead:

- **Local-first.** SQLite + FTS5 + an in-process numpy matrix. Your
  file on your disk.
- **Hybrid retrieval.** Dense cosine for meaning, BM25 for exact identifiers,
  fused with Reciprocal Rank Fusion so both channels contribute.
- **Progressive disclosure.** Search returns short previews. The agent fetches a
  full body only for the ids that turned out to matter, so recall does not cost
  you context.
- **Duplicates collapse.** Re-stating something you already stored reinforces
  the existing record instead of adding a near-identical row that competes in
  every future search.
- **Deletes are reversible until you confirm them.**

## Install

```sh
cd /Users/nsjg/Desktop/priv-projects/local_memory
uv sync
uv run local-memory doctor      # verifies the store and model agree
```

The first tool call downloads a 67 MB embedding model to
`~/.cache/local-memory/models`. `doctor` reports `model_cached: true` once it is
warm.

## Connect an agent

Both clients spawn the server on demand, so there is nothing to daemonise.

```sh
# Claude Code
claude mcp add --scope user local-memory -- \
  /Users/nsjg/Desktop/priv-projects/local_memory/.venv/bin/local-memory-stdio

# verify
claude mcp list
```

For OpenCode, add to `~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "local-memory": {
      "type": "local",
      "command": ["/Users/nsjg/Desktop/priv-projects/local_memory/.venv/bin/local-memory-stdio"],
      "enabled": true,
      "timeout": 15000
    }
  }
}
```

`timeout` is in milliseconds and defaults to 5000. The 15000 leaves headroom for
the one-off model download on first launch.

## Tools

| Tool | Purpose |
| --- | --- |
| `memory_search` | Hybrid search. Returns previews, not whole records. |
| `memory_get` | One record in full, by id. |
| `memory_store` | Create, or reinforce an existing near-duplicate. |
| `memory_forget` | Request a deletion. Pending, not immediate. |
| `memory_confirm_delete` | Approve or dismiss pending deletions. |
| `memory_stats` | Counts, active model, scope mappings. |

`kind` is an open vocabulary. Seeded with `pattern`, `decision`, `project`,
`preference`, `fact`, `lesson`, `correction`; agents may coin new ones and they
are registered automatically.

## CLI

The CLI is for you; the MCP tools are for the agent.

```sh
uv run local-memory doctor              # store/model consistency check
uv run local-memory browse              # interactive read-only viewer
uv run local-memory search "connection pooling" --detail full
uv run local-memory get <id>
uv run local-memory reindex             # required after changing model
uv run local-memory config --json
uv run local-memory models              # embedding model registry
uv run local-memory kinds               # seeded vs in-use record kinds
uv run local-memory purge --yes         # drop tombstones past retention
```

## Configuration

`~/.config/local-memory/config.toml`, created on first run.

```toml
db_path = "~/.local/share/local-memory/memory.db"
model_cache = "~/.cache/local-memory/models"
default_model = "BAAI/bge-small-en-v1.5"
threads = 3
dedup_threshold = 0.92
rrf_k = 60
tombstone_retention_days = 730

[scope.work]
paths = ["~/Desktop/VStorm"]
```

Scopes map directory prefixes to labels (`personal`, `work`, `uni`, `general`).
A scope whose path does not exist produces a warning rather than a failure, so a
moved project directory is reported, not fatal.

## Optional: HTTP transport

Only needed if you want the server reachable over TCP. It binds loopback only and
has no authentication.

```sh
./scripts/install_http_agent.sh     # LaunchAgent on 127.0.0.1:8787
./scripts/uninstall_http_agent.sh
```

## Browse

```sh
uv run local-memory browse
```

An interactive read-only table of everything the store holds. No model load, so
it starts instantly.

```
                        local-memory  ·  status=active
       TITLE                    KIND             SCOPE         CONF     ×      UPDATED
────────────────────────────────────────────────────────────────────────────────────────
▸      Connection timeouts…    fact             work          0.50            2026-10-04 20:53
       Use pgbouncer for poo…   pattern          work          0.60     1      2026-10-04 20:53
       Never force push         lesson           work          0.50            2026-10-04 20:53
1-3 of 7  ·  cursor 1
status=active   [? help] [/ search] [q quit]
```

| Key | Action |
| --- | --- |
| `j` / `k`, arrows | move |
| `g` / `G`, Home / End | first / last |
| PgDn / PgUp, `b` / `w` | page |
| Enter | full record |
| `/` | hybrid search, live as you type |
| `c` | clear search |
| `?` | help |
| `q` | quit |

Filters:

```sh
local-memory browse --scope work
local-memory browse --kind pattern --kind lesson
local-memory browse --status pending_delete --status superseded
local-memory browse --search "connection pooling"
local-memory browse -n 20
```

Two things worth knowing:

- **It never writes.** Every statement it runs is a `SELECT`, enforced by a
  test that counts non-`SELECT` statements against a real store. Deletes still
  go through the `memory_forget` → `memory_confirm_delete` flow.
- **It degrades to a plain table when piped.** Redirecting the output prints one
  table and exits rather than attempting cursor control.

Search inside `browse` only ever sees `active` records — that is what search
means everywhere else. Listing is the way to inspect `pending_delete` and
tombstones.

## Inspecting the database directly

The file is ordinary SQLite and `sqlite3` ships with macOS, so this works:

```sh
sqlite3 ~/.local/share/local-memory/memory.db \
  "select status, count(*) from records group by status;"
```

One trap: **do not open it read-only.** The database is in WAL mode, and a
read-only connection cannot create the `-shm` file WAL requires:

```
$ sqlite3 "file:$HOME/.local/share/local-memory/memory.db?mode=ro" "select 1;"
Parse error: unable to open database file (14)
```

`immutable=1` appears to work but silently ignores committed WAL transactions,
so it can show you stale data. Use `mode=ro` only if you know the `-wal` and
`-shm` files both exist. `browse` sidesteps this by opening normally and issuing
only `SELECT`s.

See [docs/GLOSSARY.md](docs/GLOSSARY.md) for what the tables, shadow tables and
sidecar files actually are.

## Design notes

- [docs/DESIGN.md](docs/DESIGN.md) — retrieval pipeline, concurrency model, and
  the tradeoffs behind each choice.
- [docs/GLOSSARY.md](docs/GLOSSARY.md) — WAL, FTS5 shadow tables, triggers and
  the other SQLite machinery that is easy to forget.

## Development

```sh
uv run pytest
uv run ruff check .
uv run ruff format .
```

260 tests, about six seconds, no network. Tests use a deterministic fake
embedder, so nothing is ever downloaded.

Two testing notes specific to this project:

- The interactive viewer cannot be fully tested in-process, because `rich.Live`
  does not repaint to a non-terminal. `tests/test_browse.py` substitutes a
  recording renderer; the real paint path was verified by driving `browse`
  through a pty.