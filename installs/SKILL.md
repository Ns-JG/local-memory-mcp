---
name: local-memory-mcp
description: >-
  Reference for the local-memory MCP server available to every opencode session.
  Use it as the long-term memory aggregation service about the user NsJG — his
  coding style and architectural decisions, personal facts, university and work
  context, people, preferences, patterns, lessons and corrections. Loaded when
  storing, searching, interviewing, or personalizing against the local-memory
  tools (memory_search, memory_get, memory_store, memory_forget,
  memory_confirm_delete, memory_stats).
---

# local-memory MCP — Long-Term Memory about NsJG

This MCP is **the memory of the user**, not a scratchpad. It stores durable
knowledge about **NsJG** gathered across every conversation and every project:

- Coding style, architectural and lexical decisions (how he writes Python, C, …)
- Domain / project knowledge and patterns that turned out to work
- Personal facts: life, university (Politechnika Wrocławska), people, plans
- Preferences, corrections, lessons learned

Everything is local (SQLite + FTS5 + embeddings), so it is private, cheap and
available in every session. Use it as a memory companion: **search it before
acting, and write into it whenever the user makes a durable decision.**

> Division of labour with other systems: context, decisions and personalization
> live **here**. Scheduling and nudging (tasks the user wants triggered later)
> belong to the *nudge-system MCP*, which does not exist yet. Do not store
> prospective task queues here — store the context those nudges will need.

## When to use this skill

- You have the `memory_search`, `memory_store`, … tools available and the user
  is asking you to remember, recall, or get to know them.
- You are in any conversation where the user makes decisions, states
  preferences, corrects you, or shares personal or project context worth
  keeping.
- You are asked to conduct an interview (e.g. about coding style) or to
  personalize your output to the user.

## The six tools, with flags and flag types

### `memory_search` — hybrid retrieval (semantic + keyword)

```text
memory_search(
    query: str,                  # required. What you are trying to remember, in natural language.
    k: int = 5,                  # how many results to return
    scope: str | None = None,    # filter: "personal" | "work" | "uni" | "general"
    kind: str | None = None,     # filter, e.g. "pattern" | "decision" | "preference"
    detail: str = "snippets",    # "titles" | "snippets" (default) | "full"
) -> str                          # JSON
```

Returns `count` and `results[]` with `id, title, gist, kind, scope, tags,
fused_score, dense_rank, lexical_rank, found_by`; `detail="snippets"` also adds
a hint to fetch full bodies only for ids that matter.

### `memory_get` — fetch a full record

```text
memory_get(
    record_id: str,              # required. id from memory_search or memory_store
) -> str                          # JSON with the full record incl. body
```

### `memory_store` — create, reinforce, or supersede

```text
memory_store(
    title: str,                  # required. short, self-contained restatement (the identity).
    body: str,                   # required. full statement, usable in a month without context.
    gist: str = "",              # optional 1-2 sentence preview; derived from body if empty.
    kind: str = "pattern",       # pattern|decision|project|preference|fact|lesson|correction, or a new one
    scope: str = "general",      # "personal" | "work" | "uni" | "general"
    tags: list[str] | None = None,   # short lowercase keywords, e.g. ["python", "testing"]
    supersedes_id: str | None = None,  # id this record replaces; old one is retired in the same transaction
    confidence: float = 0.5,     # 0.0–1.0 how well established this is
    source: str = "agent_inferred",  # e.g. "user_stated", "agent_observed"
) -> str                          # JSON {"outcome": "created"|"reinforced"|"superseded", ...}
```

### `memory_forget` — request deletion (two-step)

```text
memory_forget(
    record_id: str,              # required. the record to retire.
    reason: str = "",            # why; kept as an audit trail
) -> str                         # JSON, pending_confirmation = true
```

The record leaves search immediately but is **not deleted until confirmed**.
Never tell the user something is deleted on the strength of this call alone.

### `memory_confirm_delete` — approve or dismiss pending deletions

```text
memory_confirm_delete(
    record_ids: list[str],       # required. the pending records being decided about.
    approve: bool,               # required. True finalises deletion, False keeps them.
) -> str                          # JSON per-id outcomes
```

### `memory_stats` — inspect the store

```text
memory_stats() -> str            # no arguments. counts, active model, scope paths, warnings
```

## Scopes and kinds — how knowledge is organised

- **Scopes** map directory prefixes to labels.
  Work → `~/Desktop/VStorm/*`,
  uni → `~/Desktop/pwr_mgr/*`,
  personal → `~/Desktop/priv-projects/*`,
  `general` catches everything else. Scope is context, not importance.
- **Kinds** are an open vocabulary. Seeded: `pattern`, `decision`, `project`,
  `preference`, `fact`, `lesson`, `correction`. Coin a new kind if none fit —
  it is registered automatically.
- **Backing store**: `~/.local/share/local-memory/memory.db` (SQLite, never open
  it `mode=ro`/`immutable=1` — WAL breaks), config at
  `~/.config/local-memory/config.toml`, model cache under
  `~/.cache/local-memory/models`.

## Recalling — progressive disclosure

1. Start broad: `memory_search("did we decide anything about X", k=5)`.
2. Read only the snippets; pick the promising ids.
3. `memory_get(id)` for the full body of just those.
4. Do **not** dump whole bodies into context speculatively.

Search only sees `active` records. Before making a recommendation or starting
substantial work, search first — the answer may already exist from a past
session.

## Writing — memory as an aggregation service

This MCP is designed to accumulate. Every conversation the user has with any
agent is a potential source:

- **Durable decisions** → `kind="decision"`, note the trade-off and who decided.
- **How the user codes** → `kind="preference"` or `kind="pattern"` (see the
  interview section below).
- **Personal facts** — university, people, life events → `kind="fact"`,
  `scope="personal"` (or the matching scope).
- **Corrections** the user makes to you → `kind="correction"`, so future agents
  do not repeat the mistake.
- **Lessons** from projects → `kind="lesson"` with enough context to be useful
  later.

**Write records, not transcripts.** `body` must stand alone in a month: explain
context, not just that a thing happened. Titles must be self-contained ("BGE
defaults to 384-dim", not "this approach").

### Deduplication is automatic and is a feature

`memory_store` compares new content against existing records (0.92 cosine). A
restatement **reinforces** the existing record instead of creating a near-duplicate
— outcome is `"reinforced"`, the existing record gains confidence. That is a
success. Do not panic at `reinforced`; do not store near-identical copies.

If the existing content is now **wrong**, do not store a contradictory second
copy. Search for the record, then store with `supersedes_id` — the old record is
retired in the same transaction (outcome `"superseded"`).

### The interview pattern (intentional)

The user will sometimes ask you to **interview** them — e.g. about how programming language 
should be written. The goal is to learn his style so you can mimic it, and to
persist the *architectural and lexical decisions* he makes. This is deliberate:

1. Ask focused questions and listen for concrete signals: naming style,
   formatting, error handling, structure/preferences, tooling choices.
2. As you learn, **update the record** on the fly — search
   (`memory_search`), then `memory_store` with `supersedes_id`, or a fresh
   `memory_store` that will reinforce an existing match.
3. Turn each confirmed preference into a standalone record: e.g. `"NsJG writes
   C with snake_case functions, early returns and explicit error codes"`.
4. Prefer improving one record incrementally over creating many overlapping
   ones.

### Personal info gathering

The user will hand you facts about his life — university, people, plans — to
store for later. Capture them as `fact` records with accurate scope/tags and
enough body to be useful later. When a later session needs that context, it will
come from here.

## Deleting — two steps, nothing destroys without confirmation

`memory_forget(id, reason)` → hidden from search, `pending_confirmation: true`.
The user then decides; `memory_confirm_delete([id], approve=True|False)`
finalises or rescues the record. Report deletions as *pending* until confirmed.

## Quick checklist

- [ ] Before substantial work: `memory_search` first.
- [ ] Heard a durable decision/preference/fact? `memory_store` it.
- [ ] Restating a stored idea? Expected to `reinforce`, not duplicate.
- [ ] Existing record is wrong? `supersedes_id`, never a contradictory copy.
- [ ] User corrected you? Save a `correction` so future agents inherit it.
- [ ] Deleting? `memory_forget` then wait for `memory_confirm_delete`.
- [ ] Nudges/scheduling? Out of scope — that is the future nudge-system MCP.