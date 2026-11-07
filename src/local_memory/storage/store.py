"""Record persistence: CRUD with optimistic writes, kinds and vectors.

``rev`` rather than ``updated_at``: the timestamp has one-second resolution, so
two writes in the same second are indistinguishable and the swap would pass when
it must fail. ``rev`` is a per-row version counter; ``updated_at`` remains the
wall clock that cross-device merges compare.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from local_memory.errors import ConflictError, NotFoundError
from local_memory.models import SEARCHABLE, MemoryRecord, Status
from local_memory.storage.db import Database
from local_memory.storage.lexical import lexical_search
from local_memory.storage.schema import RECORD_COLUMNS
from local_memory.utils import new_id, utcnow


class Store(Database):
    """Every record statement. Connection and transactions live in ``Database``."""

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
            f"SELECT {', '.join(RECORD_COLUMNS)} FROM records WHERE id=?", (record_id,)
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
            f"SELECT {', '.join(RECORD_COLUMNS)} FROM records WHERE id IN ({placeholders})",
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
            f"SELECT {', '.join(RECORD_COLUMNS)} FROM records WHERE {where} "
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
        weights: Mapping[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        """BM25 over FTS5, filtered. See :func:`lexical_search`."""
        where, params = self._filter_sql(scope, kind, statuses)
        return lexical_search(self._conn, query, where, params, limit, weights)

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
        placeholders = ",".join("?" * len(RECORD_COLUMNS))
        conn.execute(
            f"INSERT INTO records ({', '.join(RECORD_COLUMNS)}) VALUES ({placeholders})",
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

        sets = ", ".join(f"{c}=?" for c in RECORD_COLUMNS if c != "id")
        params = [
            getattr(rec, c) if c != "tags" else json.dumps(rec.tags)
            for c in RECORD_COLUMNS
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
            f"SELECT {', '.join(RECORD_COLUMNS)} FROM records ORDER BY rowid"
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
