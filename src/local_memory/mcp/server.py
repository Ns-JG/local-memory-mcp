"""MCP tool surface.

Thin adapters over :class:`MemoryService`. Two rules keep these from becoming
the place where logic lives:

* No tool embeds, filters, or ranks. It unpacks arguments, calls one service
  method, and formats.
* Errors are translated to short strings the agent can act on, never raw
  tracebacks. ``ConflictError`` in particular must reach the calling agent as
  "re-read and retry" rather than as a stack trace.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from local_memory.app import build_service
from local_memory.config import Config, ConfigManager
from local_memory.errors import ConflictError, LocalMemoryError, NotFoundError, ValidationError
from local_memory.service import DETAIL_TIERS, MemoryService

if TYPE_CHECKING:
    from fastmcp import FastMCP

SERVER_NAME = "local-memory"


def instructions(config: Config) -> str:
    text = config.instructions_path.read_text(encoding="utf-8")
    return text.replace("{seed_kinds}", ", ".join(config.seed_kinds)).replace(
        "{user_name}", config.user_name
    )


def _record_payload(record: Any, *, include_body: bool = True) -> dict[str, Any]:
    return {
        "id": record.id,
        "title": record.title,
        "gist": record.gist,
        "kind": record.kind,
        "scope": record.scope,
        "tags": record.tags,
        "status": record.status,
        "confidence": record.confidence,
        "times_confirmed": record.times_confirmed,
        "content_hash": record.content_hash,
        "supersedes_id": record.supersedes_id,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "rev": record.rev,
        **({"body": record.body} if include_body else {}),
    }


def _hit_payload(hit: Any) -> dict[str, Any]:
    return {
        "id": hit.id,
        "title": hit.title,
        "gist": hit.gist,
        "kind": hit.kind,
        "scope": hit.scope,
        "tags": hit.tags,
        "fused_score": round(hit.fused_score, 6),
        "dense_rank": hit.dense_rank,
        "lexical_rank": hit.lexical_rank,
        "found_by": hit.found_by,
        **({"body": hit.body} if hit.body is not None else {}),
    }


def register_tools(mcp: Any, service: MemoryService, config: Callable[[], Config]) -> None:
    """Attach the tool set to a FastMCP server instance."""

    @mcp.tool()
    def memory_search(
        query: str,
        k: int = 5,
        scope: str | None = None,
        kind: str | None = None,
        detail: str = "snippets",
    ) -> str:
        """Search memory for records relevant to a query.

        Args:
            query: What you are trying to remember, in natural language.
            k: How many results to return.
            scope: Restrict to one scope (see `memory_stats` for the available ones).
            kind: Restrict to one kind (e.g. pattern, decision, preference).
            detail: How much of each record to return - "titles", "snippets"
                (default) or "full". Start with snippets and call `memory_get`
                for the ids that look useful.
        """
        try:
            hits = service.search(query, k=k, scope=scope, kind=kind, detail=detail)
        except (ValidationError, LocalMemoryError) as exc:
            return f"search failed: {exc}"

        payload: dict[str, Any] = {
            "count": len(hits),
            "results": [_hit_payload(h) for h in hits],
        }
        if detail == "snippets" and hits:
            payload["hint"] = "Previews only. Call memory_get with an id for the full body."
        warnings = config().scope_warnings()
        if warnings:
            payload["scope_warnings"] = warnings
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @mcp.tool()
    def memory_get(record_id: str) -> str:
        """Fetch one record in full, including its body.

        Args:
            record_id: The id returned by `memory_search` or `memory_store`.
        """
        try:
            record = service.get(record_id)
        except NotFoundError:
            return f"no record with id {record_id!r}"
        except LocalMemoryError as exc:
            return f"get failed: {exc}"
        return json.dumps(_record_payload(record), indent=2, ensure_ascii=False)

    @mcp.tool()
    def memory_store(
        title: str,
        body: str,
        gist: str = "",
        kind: str = "pattern",
        scope: str | None = None,
        tags: Sequence[str] | None = None,
        supersedes_id: str | None = None,
        confidence: float = 0.5,
        source: str = "agent_inferred",
    ) -> str:
        """Store a durable fact, pattern, decision, preference or lesson.

        Args:
            title: A short, self-contained restatement. This is the record's
                identity and the thing that will be matched later, so do not
                write "the above" or "this approach".
            body: The full statement, with enough context to still be correct
                and useful in a month without this conversation.
            gist: Optional one-or-two-sentence preview. Derived from the body
                if omitted; pass one when you can write it better.
            kind: pattern, decision, project, preference, fact, lesson,
                correction - or a new kind if none of those fit.
            scope: One of the configured scopes; omit to use the default.
            tags: Short lowercase keywords, e.g. ["python", "testing"].
            supersedes_id: Set when this record replaces an existing one. The
                old record is retired in the same transaction; do not store
                both versions side by side.
            confidence: 0.0-1.0 based on how well established this is.
            source: Where the knowledge came from, e.g. "user_stated".

        Returns the outcome: "created", "reinforced" or "superseded".
        """
        try:
            record, outcome = service.store_record(
                title=title,
                body=body,
                gist=gist,
                kind=kind,
                scope=scope,
                tags=list(tags or []),
                supersedes_id=supersedes_id,
                confidence=confidence,
                source=source,
            )
        except NotFoundError as exc:
            return f"supersedes_id is not a known record: {exc}"
        except ConflictError as exc:
            return f"conflict: {exc}. Re-read the record with memory_get and retry."
        except (ValidationError, LocalMemoryError) as exc:
            return f"store failed: {exc}"

        note = {
            "created": "Stored.",
            "reinforced": (
                "This restates an existing record, so the existing one was "
                "reinforced rather than duplicated."
            ),
            "superseded": "Stored and the record it replaces is now retired.",
        }[outcome]
        return json.dumps(
            {"outcome": outcome, "note": note, "record": _record_payload(record)},
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def memory_forget(record_id: str, reason: str = "") -> str:
        """Request deletion of a record. Pending confirmation, not immediate.

        The record stops appearing in search results now, but nothing is
        destroyed until the user confirms. Tell the user a deletion is pending, not
        that it is done.

        Args:
            record_id: The record to retire.
            reason: Why it is being forgotten, kept as an audit trail.
        """
        try:
            record = service.forget(record_id, reason=reason)
        except NotFoundError:
            return f"no record with id {record_id!r}"
        except (ValidationError, LocalMemoryError) as exc:
            return f"forget failed: {exc}"
        return json.dumps(
            {
                "status": record.status,
                "pending_confirmation": True,
                "note": (
                    f"Removed from search. {config().user_name} must confirm before it is "
                    "actually deleted. Do not report it as deleted."
                ),
                "record": _record_payload(record),
            },
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def memory_confirm_delete(record_ids: Sequence[str], approve: bool) -> str:
        """Approve or dismiss pending deletions, as instructed by the user.

        Args:
            record_ids: The pending records the user is deciding about.
            approve: True to finalise the deletion, False to keep the records.
        """
        try:
            results = service.confirm_delete(list(record_ids), approve=approve)
        except NotFoundError as exc:
            return f"no such record: {exc}"
        except ConflictError as exc:
            return f"conflict: {exc}. Re-read with memory_get and retry."
        except LocalMemoryError as exc:
            return f"confirm failed: {exc}"
        return json.dumps(
            {"approved": approve, "results": [{"id": i, "outcome": o} for i, o in results]},
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def memory_stats() -> str:
        """Counts, active embedding model, and configured scope mappings."""
        stats = service.stats()
        cfg = config()
        stats["scopes"] = {
            name: [str(p) for p in cfg.scopes.rules[name].paths] if name in cfg.scopes.rules else []
            for name in cfg.scopes.available
        }
        stats["default_scope"] = cfg.scopes.default
        stats["scope_warnings"] = cfg.scope_warnings()
        return json.dumps(stats, indent=2, ensure_ascii=False)


def build_server(
    service: MemoryService | None = None, manager: ConfigManager | None = None
) -> FastMCP:
    """A FastMCP server with the tool set attached."""
    from fastmcp import FastMCP

    manager = manager or ConfigManager()
    mcp = FastMCP(name=SERVER_NAME, instructions=instructions(manager.load()))
    register_tools(mcp, service or build_service(manager), manager.load)
    return mcp


__all__ = ["DETAIL_TIERS", "SERVER_NAME", "build_server", "instructions", "register_tools"]
