# local-memory MCP — Global Agent Instructions

<!-- Rendered from installs/mcp_instructions.md by scripts/install_mcp_instructions.sh; edit the template, not the installed copy. -->

This MCP provides a **local-first, hybrid-retrieval memory service** for coding agents.
It stores durable facts, patterns, decisions, preferences, and lessons in a SQLite database
with FTS5 full-text search and vector embeddings, accessible via 6 MCP tools. The memories are about {{USER_NAME}}.

## What This MCP Does

- **Stores** your coding patterns, architectural decisions, preferences, and lessons
- **Searches** via hybrid retrieval (semantic embeddings + BM25 keyword) — finds paraphrases AND exact identifiers
- **Deduplicates** automatically at 0.92 cosine similarity (reinforces existing records instead of creating near-duplicates)
- **Keeps project notes** across sessions (state, decisions, ideas) — only with {{USER_NAME}}'s explicit approval, see *Project Notes & Ideas*
- **Tracks** record lifecycle: active → pending_delete → deleted (with confirmation), or superseded
- **Scopes** records by project: {{AVAILABLE_SCOPES}} (mapped from directory paths)
- **Works offline** — no network at runtime except one-time model download

## When to Use This Memory

**Store when the user explicitly asks, or when you observe:**
- A decision made ("We'll use pgbouncer for pooling")
- A pattern established ("Ruff line length is 100")
- A lesson learned ("Never force push to main")
- A preference stated ("I prefer table-driven tests")
- A fact relevant to future work ("The release deadline is June")
- A correction ("Actually, use `uv` not `pip`")

Project and idea notes are the exception: store those only after the user approves (see *Project Notes & Ideas*).

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
- `scope`: {{AVAILABLE_SCOPES}} — restricts to project context
- `kind`: {{SEED_KINDS_BAR}} — or any custom kind
- `detail`: `"titles" | "snippets" | "full"` — how much to return

### Store (Deduplication is Automatic)
```python
memory_store(
    title="Use pgbouncer for pooling",
    body="Postgres connection pooling runs through pgbouncer in transaction mode.",
    kind="pattern",
    scope="{{DEFAULT_SCOPE}}",
    tags=["db", "perf"],
    source="user_stated",
    confidence=0.9,
)
```
- If a record at ≥0.92 cosine exists, it reinforces (confidence↑, tags merge) instead of creating duplicate
- To replace a wrong record: pass `supersedes_id` — old record marked `superseded` in same transaction

### Delete is Two-Step
1. `memory_forget(id, reason="superseded by better approach")` → record becomes `pending_delete`, hidden from search
2. User confirms: `memory_confirm_delete([id], approve=true)` → permanently deleted
3. Never tell user "deleted" after `memory_forget` alone — it's pending

## Project Notes & Ideas

The memory can also hold working notes about a project or an idea, so the next session — in any agent — can pick up where this one stopped. The notes belong to {{USER_NAME}}: **store them only with their explicit approval, every time.** Never write project notes on your own initiative.

- **What to keep:** current state, decisions and why, open questions, next steps, ideas that came up and were not acted on.
- **How:** `kind="project"` (or `"idea"`), the scope that matches the project's directory, a tag with the project name, a self-contained title (`"<project>: <what this note covers>"`), `source="user_approved"`.
- **Update, don't pile up:** `memory_search("<project name>")` first and use `supersedes_id` on the project's existing note instead of adding another.
- **Resume:** when starting work in a known project, search for its notes before doing anything else.

### For the agent only: offer to save before compaction

When the context window is close to full, the harness warns that compaction is coming, the user is about to `/compact` or end the session, or a stretch of project work or new ideas has not been saved yet — **offer, do not store.** Tell the user that now is the time, and what you would save, for example:

> This session is close to compaction, so now is a good time to save notes about <project / idea> to memory, if you'd like. I'd keep: (1) …, (2) …. Want me to store them?

Store only after a yes. After a no, drop it and do not ask again for the same notes.

## Scopes & Kinds

**Scopes** (config-driven, mapped from directory prefixes):
{{SCOPE_MAPPINGS}}

**Kinds** (open vocabulary, seeded with):
{{SEED_KINDS}} — coin new ones as needed

## Configuration & Storage

| File | Purpose |
|------|---------|
| `{{CONFIG_PATH}}` | Scopes, model, thresholds |
| `{{DB_PATH}}` | SQLite + FTS5 + vectors (WAL mode) |
| `{{MODEL_CACHE}}` | Embedding model cache (`{{DEFAULT_MODEL}}`) |

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
| "Save notes about this project" (after approval) | `memory_store(title, body, kind="project", scope, tags=[project])` |
| "That's wrong, delete it" | `memory_forget(id, reason)` → user confirms |
| "What's in the DB?" | `memory_stats()` |