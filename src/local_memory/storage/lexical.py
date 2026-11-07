"""FTS5 lexical search: query sanitising and BM25 ranking."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any

from local_memory.storage.schema import FTS_COLUMNS

_FTS_TOKEN: re.Pattern[str] = re.compile(r"[\w*]+", re.UNICODE)


def _fts_tokens(query: str) -> list[str]:
    """Free text as FTS5 phrase literals."""
    return _FTS_TOKEN.findall(query)


def _quote(tokens: Sequence[str]) -> str:
    return " ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def sanitize_fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Every token is double-quoted, which makes FTS5 treat it as a literal and
    turns the sequence into an implicit AND. Without this, a query containing
    ``"``, ``*`` or ``NEAR`` is a syntax error at best and a query that escapes
    its filter context at worst.
    """
    return _quote(_fts_tokens(query))


def lexical_search(
    conn: sqlite3.Connection,
    query: str,
    where: str,
    params: Sequence[Any],
    limit: int,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[str, float]]:
    """BM25 over FTS5, filtered by ``where``. Returns ``(id, score)``, best first.

    Two passes. The first is implicit AND across all query tokens - highest
    precision, and the reason a rare term like ``pgbouncer`` ranks first.
    But AND is brittle: querying ``postgresql connection`` against a record
    that says ``postgres`` matches nothing at all, silently discarding the
    lexical channel for an ordinary question. So if AND returns fewer than
    ``limit`` rows, a second OR pass tops the list up, with AND results
    keeping precedence.

    ``weights`` maps every FTS column to its BM25 weight; omitted means equal weights.

    FTS5's ``bm25()`` returns negative values where more negative is better,
    so scores are negated here to give a conventional "higher is better".
    """
    tokens = _fts_tokens(query)
    if not tokens or limit <= 0:
        return []

    strict = _fts_hits(conn, _quote(tokens), where, params, limit, weights)
    if len(strict) >= limit or len(tokens) == 1:
        return strict

    loose = _fts_hits(conn, " OR ".join(f'"{t}"' for t in tokens), where, params, limit, weights)
    merged: list[tuple[str, float]] = []
    seen: set[str] = set()
    for record_id, score in strict + loose:
        if record_id not in seen:
            seen.add(record_id)
            merged.append((record_id, score))
        if len(merged) == limit:
            break
    return merged


def _fts_hits(
    conn: sqlite3.Connection,
    match: str,
    where: str,
    params: Sequence[Any],
    limit: int,
    weights: Mapping[str, float] | None,
) -> list[tuple[str, float]]:
    ordered = [weights[c] for c in FTS_COLUMNS] if weights else []
    rows = conn.execute(
        f"""
        SELECT r.id AS id, bm25(records_fts{", ?" * len(ordered)}) AS score
        FROM records_fts
        JOIN records r ON r.rowid = records_fts.rowid
        WHERE records_fts MATCH ? AND {where}
        ORDER BY score
        LIMIT ?
        """,
        (*ordered, match, *params, limit),
    ).fetchall()
    return [(r["id"], -float(r["score"])) for r in rows]
