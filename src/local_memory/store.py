"""SQLite persistence: schema, FTS5 lexical index, vectors, optimistic writes.

Concurrency model, in three layers:
1. In-process: a re-entrant lock serialises write transactions. 
    Reads run lock-free under WAL.
2. Cross-process: ``busy_timeout`` plus full-jitter retry on lock contention.
3. Logical conflict: every update carries ``WHERE rev = ?``. A rowcount of zero
   means a concurrent writer won the race, which raises :class:`ConflictError`
   rather than silently clobbering their write.

``rev`` rather than ``updated_at``: the timestamp has one-second resolution, so
two writes in the same second are indistinguishable and the swap would pass when
it must fail. ``rev`` is a per-row version counter; ``updated_at`` remains the
wall clock that cross-device merges compare.
"""

from __future__ import annotations

import contextlib
import json
import random
import re
import sqlite3
import time
from collections.abc import Generator, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .models import (
    SCHEMA_VERSION,
    SEARCHABLE,
    ConflictError,
    LocalMemoryError,
    MemoryRecord,
    NotFoundError,
    Scope,
    Status,
    new_id,
    utcnow,
)

# FTS5 column weights, title dominates as it is the compressed inentity of a record.
BM25_WEIGHTS: dict[str, float] = {
    "title": 10.0,
    "gist": 5.0,
    "body": 1.0,
    "tags": 3.0
}

_FTS_TOKEN: re.Pattern[str] = re.compile(r"[\w*]+", re.UNICODE)

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
  rowid           INTEGER PRIMARY KEY,
  id              TEXT NOT NULL UNIQUE,
  kind            TEXT NOT NULL,
  scope           TEXT NOT NULL,
  title           TEXT NOT NULL,
  gist            TEXT NOT NULL,
  body            TEXT NOT NULL,
  tags            TEXT NOT NULL DEFAULT '[]',
  status          TEXT NOT NULL DEFAULT 'active',
  supersedes_id   TEXT,
  delete_reason   TEXT,
  confidence      REAL NOT NULL DEFAULT 0.5,
  source          TEXT NOT NULL,
  times_confirmed INTEGER NOT NULL DEFAULT 0,
  content_hash    TEXT NOT NULL,
  embedding_model TEXT,
  embedding_dim   INTEGER,
  embedded_at     TEXT,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  deleted_at      TEXT,
  origin_device   TEXT,
  rev             INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_records_scope   ON records(scope);
CREATE INDEX IF NOT EXISTS idx_records_kind    ON records(kind);
CREATE INDEX IF NOT EXISTS idx_records_status  ON records(status);
CREATE INDEX IF NOT EXISTS idx_records_updated ON records(updated_at);
CREATE INDEX IF NOT EXISTS idx_records_super   ON records(supersedes_id);

CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
  title, gist, body, tags,
  content='records',
  content_rowid='rowid',
  tokenize="unicode61 remove_diacritics 2 tokenchars '_'"
);

CREATE TRIGGER IF NOT EXISTS records_ai AFTER INSERT ON records BEGIN
  INSERT INTO records_fts(rowid, title, gist, body, tags)
  VALUES (new.rowid, new.title, new.gist, new.body, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS records_ad AFTER DELETE ON records BEGIN
  INSERT INTO records_fts(records_fts, rowid, title, gist, body, tags)
  VALUES ('delete', old.rowid, old.title, old.gist, old.body, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS records_au AFTER UPDATE ON records BEGIN
  INSERT INTO records_fts(records_fts, rowid, title, gist, body, tags)
  VALUES ('delete', old.rowid, old.title, old.gist, old.body, old.tags);
  INSERT INTO records_fts(rowid, title, gist, body, tags)
  VALUES (new.rowid, new.title, new.gist, new.body, new.tags);
END;

CREATE TABLE IF NOT EXISTS vectors (
  id  TEXT PRIMARY KEY REFERENCES records(id),
  vec BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS kinds (
  name       TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  note       TEXT
);

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);
"""

_RECORD_COLUMNS = (
    "id",
    "kind",
    "scope",
    "title",
    "gist",
    "body",
    "tags",
    "status",
    "supersedes_id",
    "delete_reason",
    "confidence",
    "source",
    "times_confirmed",
    "content_hash",
    "embedding_model",
    "embedding_dim",
    "embedded_at",
    "created_at",
    "updated_at",
    "deleted_at",
    "origin_device",
    "rev",
)


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


class Store:
    """Owns the SQLite connection and all statements."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        max_retries: int = 5,
        retry_base: float = 0.025,
    ) -> None:
        self.db_path = Path(db_path)
        self._max_retries = max_retries
        self._retry_base = retry_base
        self._write_lock = __import__("threading").RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,
            timeout=5.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._migrate()

    def _configure(self) -> None:
        cur = self._conn
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")

    def _migrate(self) -> None:
        with self._write_lock:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise LocalMemoryError(
                    f"db schema version {version} is newer than supported {SCHEMA_VERSION}"
                )
            if version == 0:
                self._conn.executescript(SCHEMA)
                self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            elif version < SCHEMA_VERSION:
                raise LocalMemoryError(
                    f"no migration path from schema version {version} to {SCHEMA_VERSION}"
                )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- transactions ------------------------------------------------------

    @contextlib.contextmanager
    def _write(self) -> Generator[sqlite3.Connection, None, None]:
        """Run a write transaction, retrying on lock contention.

        Full jitter rather than equal jitter: with several waiters, equal jitter
        keeps them colliding in lockstep.
        """
        with self._write_lock:
            last: Exception | None = None
            for attempt in range(self._max_retries):
                try:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        yield self._conn
                    except BaseException:
                        with contextlib.suppress(sqlite3.Error):
                            self._conn.rollback()
                        raise
                    self._conn.execute(
                        "INSERT INTO meta(key, value) VALUES('store_epoch', '1') "
                        "ON CONFLICT(key) DO UPDATE SET "
                        "value = CAST(meta.value AS INTEGER) + 1"
                    )
                    self._conn.commit()
                    return
                except sqlite3.OperationalError as exc:
                    msg = str(exc).lower()
                    if "locked" not in msg and "busy" not in msg:
                        raise
                    with contextlib.suppress(sqlite3.Error):
                        self._conn.rollback()
                    last = exc
                    time.sleep(random.uniform(0, self._retry_base * (2**attempt)))
            raise LocalMemoryError(f"write failed after {self._max_retries} attempts: {last}")

    def epoch(self) -> int:
        """Monotonic write counter, bumped on every commit.

        Readers compare it to decide whether their in-memory index is stale.
        """
        row = self._conn.execute("SELECT value FROM meta WHERE key='store_epoch'").fetchone()
        return int(row[0]) if row else 0

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    # -- reads -------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            kind=row["kind"],
            scope=row["scope"],
            title=row["title"],
            gist=row["gist"],
            body=row["body"],
            tags=json.loads(row["tags"]),
            status=row["status"],
            supersedes_id=row["supersedes_id"],
            delete_reason=row["delete_reason"],
            confidence=row["confidence"],
            source=row["source"],
            times_confirmed=row["times_confirmed"],
            content_hash=row["content_hash"],
            embedding_model=row["embedding_model"],
            embedding_dim=row["embedding_dim"],
            embedded_at=row["embedded_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            deleted_at=row["deleted_at"],
            origin_device=row["origin_device"],
            rev=row["rev"],
        )

    def get(self, record_id: str) -> MemoryRecord:
        row = self._conn.execute(
            f"SELECT {', '.join(_RECORD_COLUMNS)} FROM records WHERE id=?", (record_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(f"no record with id {record_id!r}")
        return self._row_to_record(row)

    def find(self, record_id: str) -> MemoryRecord | None:
        try:
            return self.get(record_id)
        except NotFoundError:
            return None

    def get_many(self, ids: Sequence[str]) -> list[MemoryRecord]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT {', '.join(_RECORD_COLUMNS)} FROM records WHERE id IN ({placeholders})",
            tuple(ids),
        ).fetchall()
        by_id = {r["id"]: self._row_to_record(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    @staticmethod
    def _filter_sql(
        scope: str | None,
        kind: str | None,
        statuses: Sequence[str] | None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if statuses is not None:
            clauses.append(f"status IN ({','.join('?' * len(statuses))})")
            params.extend(statuses)
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        return (" AND ".join(clauses) if clauses else "1=1"), params

    def list_records(
        self,
        *,
        scope: str | None = None,
        kind: str | None = None,
        statuses: Sequence[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        where, params = self._filter_sql(scope, kind, statuses)
        rows = self._conn.execute(
            f"SELECT {', '.join(_RECORD_COLUMNS)} FROM records WHERE {where} "
            "ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def candidate_ids(
        self,
        *,
        scope: str | None = None,
        kind: str | None = None,
        statuses: Sequence[str] = SEARCHABLE,
    ) -> list[str]:
        """Ids matching a filter, in insertion order, for index-side masking."""
        where, params = self._filter_sql(scope, kind, statuses)
        rows = self._conn.execute(
            f"SELECT id FROM records WHERE {where} ORDER BY rowid", params
        ).fetchall()
        return [r[0] for r in rows]

    def count(self, *, statuses: Sequence[str] | None = None) -> int:
        where, params = self._filter_sql(None, None, statuses)
        return int(
            self._conn.execute(f"SELECT COUNT(*) FROM records WHERE {where}", params).fetchone()[0]
        )

    def lexical_search(
        self,
        query: str,
        *,
        scope: str | None = None,
        kind: str | None = None,
        statuses: Sequence[str] = SEARCHABLE,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """BM25 over FTS5, filtered. Returns ``(id, score)``, best first.

        Two passes. The first is implicit AND across all query tokens - highest
        precision, and the reason a rare term like ``pgbouncer`` ranks first.
        But AND is brittle: querying ``postgresql connection`` against a record
        that says ``postgres`` matches nothing at all, silently discarding the
        lexical channel for an ordinary question. So if AND returns fewer than
        ``limit`` rows, a second OR pass tops the list up, with AND results
        keeping precedence.

        FTS5's ``bm25()`` returns negative values where more negative is better,
        so scores are negated here to give a conventional "higher is better".
        """
        tokens = _fts_tokens(query)
        if not tokens or limit <= 0:
            return []
        where, params = self._filter_sql(scope, kind, statuses)

        strict = self._fts_hits(_quote(tokens), where, params, limit)
        if len(strict) >= limit or len(tokens) == 1:
            return strict

        loose = self._fts_hits(" OR ".join(f'"{t}"' for t in tokens), where, params, limit)
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
        self, match: str, where: str, params: Sequence[Any], limit: int
    ) -> list[tuple[str, float]]:
        weights = ", ".join(str(w) for w in BM25_WEIGHTS.values())
        rows = self._conn.execute(
            f"""
            SELECT r.id AS id, bm25(records_fts, {weights}) AS score
            FROM records_fts
            JOIN records r ON r.rowid = records_fts.rowid
            WHERE records_fts MATCH ? AND {where}
            ORDER BY score
            LIMIT ?
            """,
            (match, *params, limit),
        ).fetchall()
        return [(r["id"], -float(r["score"])) for r in rows]

    # -- writes ------------------------------------------------------------

    def _prepare_insert(self, rec: MemoryRecord) -> tuple[Any, ...]:
        rec.validate()
        if not rec.id:
            rec.id = new_id()
        now = utcnow()
        rec.created_at = rec.created_at or now
        rec.updated_at = now
        rec.rev = max(rec.rev, 1)
        rec.content_hash = rec.compute_hash()
        return (
            rec.id,
            rec.kind,
            rec.scope,
            rec.title,
            rec.gist,
            rec.body,
            json.dumps(rec.tags),
            rec.status,
            rec.supersedes_id,
            rec.delete_reason,
            rec.confidence,
            rec.source,
            rec.times_confirmed,
            rec.content_hash,
            rec.embedding_model,
            rec.embedding_dim,
            rec.embedded_at,
            rec.created_at,
            rec.updated_at,
            rec.deleted_at,
            rec.origin_device,
            rec.rev,
        )

    def _do_insert(self, conn: sqlite3.Connection, values: tuple[Any, ...]) -> None:
        placeholders = ",".join("?" * len(_RECORD_COLUMNS))
        conn.execute(
            f"INSERT INTO records ({', '.join(_RECORD_COLUMNS)}) VALUES ({placeholders})",
            values,
        )
        conn.execute(
            "INSERT OR IGNORE INTO kinds(name, created_at) VALUES (?, ?)",
            (values[1], utcnow()),
        )

    def insert(self, rec: MemoryRecord) -> MemoryRecord:
        values = self._prepare_insert(rec)
        with self._write() as conn:
            self._do_insert(conn, values)
        return rec

    def insert_and_supersede(
        self,
        rec: MemoryRecord,
        superseded_id: str,
        expected_rev: int,
    ) -> MemoryRecord:
        """Insert a successor and retire its predecessor in one transaction.

        Both rows flip together or neither does. Doing this as two writes would
        leave a window where the old record is retired but the new one is not
        yet visible - exactly the contradiction this system exists to avoid.
        """
        values = self._prepare_insert(rec)
        now = utcnow()
        with self._write() as conn:
            cur = conn.execute(
                "UPDATE records SET status=?, updated_at=?, rev=rev+1 WHERE id=? AND rev=?",
                (str(Status.SUPERSEDED), now, superseded_id, expected_rev),
            )
            if cur.rowcount == 0:
                if conn.execute("SELECT 1 FROM records WHERE id=?", (superseded_id,)).fetchone():
                    raise ConflictError(superseded_id)
                raise NotFoundError(f"no record with id {superseded_id!r}")
            self._do_insert(conn, values)
        return rec

    def update(
        self, rec: MemoryRecord, expected_rev: int, *, bump_confirmed: bool = False
    ) -> MemoryRecord:
        """Compare-and-swap an existing record.

        ``expected_rev`` is the caller's read of the row version. A rowcount of
        zero means someone else committed in between, which is a
        :class:`ConflictError` rather than a silent overwrite.
        """
        rec.validate()
        rec.content_hash = rec.compute_hash()
        rec.updated_at = utcnow()
        rec.rev = expected_rev + 1
        if bump_confirmed:
            rec.times_confirmed += 1

        sets = ", ".join(f"{c}=?" for c in _RECORD_COLUMNS if c != "id")
        params = [
            getattr(rec, c) if c != "tags" else json.dumps(rec.tags)
            for c in _RECORD_COLUMNS
            if c != "id"
        ]
        with self._write() as conn:
            cur = conn.execute(
                f"UPDATE records SET {sets} WHERE id=? AND rev=?",
                (*params, rec.id, expected_rev),
            )
            if cur.rowcount == 0:
                if conn.execute("SELECT 1 FROM records WHERE id=?", (rec.id,)).fetchone():
                    raise ConflictError(rec.id)
                raise NotFoundError(f"no record with id {rec.id!r}")
        return rec

    def set_status(
        self,
        record_id: str,
        status: str,
        *,
        expected_rev: int | None = None,
        reason: str | None = None,
    ) -> MemoryRecord:
        """Transition status with the same CAS guarantee as :meth:`update`."""
        now = utcnow()
        deleted_at = now if status == Status.DELETED else None
        with self._write() as conn:
            if expected_rev is None:
                cur = conn.execute(
                    "UPDATE records SET status=?, delete_reason=?, deleted_at=?, "
                    "updated_at=?, rev=rev+1 WHERE id=?",
                    (str(status), reason, deleted_at, now, record_id),
                )
            else:
                cur = conn.execute(
                    "UPDATE records SET status=?, delete_reason=?, deleted_at=?, "
                    "updated_at=?, rev=rev+1 WHERE id=? AND rev=?",
                    (str(status), reason, deleted_at, now, record_id, expected_rev),
                )
            if cur.rowcount == 0:
                if conn.execute("SELECT 1 FROM records WHERE id=?", (record_id,)).fetchone():
                    raise ConflictError(record_id)
                raise NotFoundError(f"no record with id {record_id!r}")
        return self.get(record_id)

    def set_embedding_info(self, record_id: str, model: str, dim: int) -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE records SET embedding_model=?, embedding_dim=?, embedded_at=?, "
                "rev=rev+1 WHERE id=?",
                (model, dim, utcnow(), record_id),
            )

    def hard_delete(self, record_id: str) -> None:
        """Physically remove a row. Only for expired tombstones.

        There is no undo. The ``records_ad`` FTS trigger drops the lexical
        entry and the foreign key takes the vector with it.
        """
        with self._write() as conn:
            conn.execute("DELETE FROM vectors WHERE id=?", (record_id,))
            conn.execute("DELETE FROM records WHERE id=?", (record_id,))

    def list_kinds(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT name FROM kinds ORDER BY name").fetchall()]

    def add_kind(self, name: str, note: str | None = None) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO kinds(name, created_at, note) VALUES (?, ?, ?)",
                (name, utcnow(), note),
            )

    def stale_records(self, model: str) -> list[str]:
        """Ids whose vectors are missing or were made by a different model."""
        return [
            r[0]
            for r in self._conn.execute(
                """
                SELECT r.id FROM records r
                LEFT JOIN vectors v ON v.id = r.id
                WHERE r.embedding_model IS NULL
                   OR r.embedding_model != ?
                   OR v.vec IS NULL
                """,
                (model,),
            ).fetchall()
        ]

    def all_records(self) -> list[MemoryRecord]:
        rows = self._conn.execute(
            f"SELECT {', '.join(_RECORD_COLUMNS)} FROM records ORDER BY rowid"
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # -- vectors -----------------------------------------------------------

    def put_vector(self, record_id: str, vec: np.ndarray) -> None:
        blob = np.asarray(vec, dtype="<f4").tobytes()
        with self._write() as conn:
            conn.execute(
                "INSERT INTO vectors(id, vec) VALUES(?, ?) "
                "ON CONFLICT(id) DO UPDATE SET vec=excluded.vec",
                (record_id, blob),
            )

    def get_vector(self, record_id: str) -> np.ndarray | None:
        row = self._conn.execute("SELECT vec FROM vectors WHERE id=?", (record_id,)).fetchone()
        return None if row is None else np.frombuffer(row[0], dtype="<f4")

    def load_vectors(self, dim: int) -> tuple[list[str], np.ndarray]:
        """Every stored vector as one matrix. Shape ``(0, dim)`` when empty."""
        rows = self._conn.execute("SELECT id, vec FROM vectors ORDER BY rowid").fetchall()
        ids = [r[0] for r in rows]
        if not rows:
            return [], np.zeros((0, dim), dtype=np.float32)
        return ids, np.vstack([np.frombuffer(r[1], dtype="<f4") for r in rows])

    def clear_vectors(self) -> None:
        with self._write() as conn:
            conn.execute("DELETE FROM vectors")


def scope_names() -> Iterable[str]:
    return tuple(s.value for s in Scope)
