"""SQLite connection, write transactions and the cross-process epoch.

Concurrency model, in three layers:
1. In-process: a re-entrant lock serialises write transactions.
   Reads run lock-free under WAL.
2. Cross-process: ``busy_timeout`` plus full-jitter retry on lock contention.
3. Logical conflict: every update carries ``WHERE rev = ?`` (see ``Store``). A rowcount
   of zero means a concurrent writer won the race, which raises ``ConflictError``
   rather than silently clobbering their write.
"""

from __future__ import annotations

import contextlib
import random
import sqlite3
import threading
import time
from collections.abc import Generator
from pathlib import Path

from local_memory.config import schema_version
from local_memory.errors import LocalMemoryError
from local_memory.storage.schema import migrate_to_schema


class Database:
    """Owns the connection. ``Store`` builds the record statements on top."""

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
        self._write_lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,
            timeout=5.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._configure()
        with self._write_lock:
            migrate_to_schema(self._conn, schema_version())

    def _configure(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextlib.contextmanager
    def _write(self) -> Generator[sqlite3.Connection]:
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
