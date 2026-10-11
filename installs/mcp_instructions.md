# local-memory MCP — Global Agent Instructions

This MCP provides a **local-first, hybrid-retrieval memory service** for coding agents.
It stores durable facts, patterns, decisions, preferences, and lessons in a SQLite database
with FTS5 full-text search and vector embeddings, accessible via 6 MCP tools.

## What This MCP Does

- **Stores** your coding patterns, architectural decisions, preferences, and lessons
- **Searches** via hybrid retrieval (semantic embeddings + BM25 keyword) — finds paraphrases AND exact identifiers
- **Deduplicates** automatically at 0.92 cosine similarity (reinforces existing records instead of creating near-duplicates)
- **Tracks** record lifecycle: active → pending_delete → deleted (with confirmation), or superseded
- **Scopes** records by project: `work`, `uni`, `personal`, `general` (mapped from directory paths)
- **Works offline** — no network at runtime except one-time model download

## When to Use This Memory

**Store when the user explicitly asks, or when you observe:**
- A decision made ("We'll use pgbouncer for pooling")
- A pattern established ("Ruff line length is 100")
- A lesson learned ("Never force push to main")
- A preference stated ("I prefer table-driven tests")
- A fact relevant to future work ("PWr deadline is June")
- A correction ("Actually, use `uv` not `pip`")

**Search when:**
- Starting a non-trivial task ("Check if we did something like that before")
- You need context from past work
- The user references something from earlier

## The 6 MCP Tools

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `memory_search` | Hybrid search — returns snippets by default | `query` (required), `k`, `scope`, `kind`, `detail` |
| `memory_get` | Fetch full record by ID | `record_id` |
| `memory_store` | Store new record (auto-dedup) | `title`, `body`, `kind`, `scope`, `tags`, `source`, `confidence`, `supersedes_id` |
| `memory_forget` | Request deletion (pending confirmation) | `record_id`, `reason` |
| `memory_confirm_delete` | Approve/reject pending deletions | `record_ids[]`, `approve` (bool) |
| `memory_stats` | Counts, active model, scope mappings | (none) |

## Tool Usage Patterns

### Search → Get (Progressive Disclosure)
```python
# 1. Broad search
hits = memory_search("connection pooling", k=5)
# 2. Fetch full bodies for relevant IDs
full = memory_get(hits[0].id)
```

### Filters
- `scope`: `"work" | "uni" | "personal" | "general"` — restricts to project context
- `kind`: `"pattern" | "decision" | "fact" | "lesson" | "correction" | "preference" | "project"` — or any custom kind
- `detail`: `"titles" | "snippets" | "full"` — how much to return

### Store (Deduplication is Automatic)
```python
memory_store(
    title="Use pgbouncer for pooling",
    body="Postgres connection pooling runs through pgbouncer in transaction mode.",
    kind="pattern",
    scope="work",
    tags=["db", "perf"],
    source="user_stated",
    confidence=0.9
)
```
- If a record at ≥0.92 cosine exists, it reinforces (confidence↑, tags merge) instead of creating duplicate
- To replace a wrong record: pass `supersedes_id` — old record marked `superseded` in same transaction

### Delete is Two-Step
1. `memory_forget(id, reason="superseded by better approach")` → record becomes `pending_delete`, hidden from search
2. User confirms: `memory_confirm_delete([id], approve=true)` → permanently deleted
3. Never tell user "deleted" after `memory_forget` alone — it's pending

## Scopes & Kinds

**Scopes** (config-driven, mapped from directory prefixes):
- `work` → `~/Desktop/VStorm/*`
- `uni` → `~/Desktop/pwr_mgr/*`
- `personal` → `~/Desktop/priv-projects/*`
- `general` → catch-all when no path matches

**Kinds** (open vocabulary, seeded with):
`pattern`, `decision`, `fact`, `lesson`, `correction`, `preference`, `project` — coin new ones as needed

## Configuration & Storage

| File | Purpose |
|------|---------|
| `~/.config/local-memory/config.toml` | Scopes, model, thresholds (created on first run) |
| `~/.local/share/local-memory/memory.db` | SQLite + FTS5 + vectors (WAL mode) |
| `~/.cache/local-memory/models/` | Embedding model cache (BAAI/bge-small-en-v1.5, 384-dim) |

**Important:** The DB uses WAL mode. Do NOT open it `mode=ro` or `immutable=1` — it will fail or show stale data. Use normal connection and issue only SELECTs.

## Server Command

The MCP runs via stdio (spawned per session by the client):
```bash
local-memory-stdio
# or from source:
uv run local-memory-stdio
```

## Quick Reference Card

| Task | Call |
|------|------|
| "What did we decide about X?" | `memory_search("X")` |
| "Show me the full record" | `memory_get(id)` |
| "Remember this pattern" | `memory_store(title, body, kind, scope)` |
| "That's wrong, delete it" | `memory_forget(id, reason)` → user confirms |
| "What's in the DB?" | `memory_stats()` |