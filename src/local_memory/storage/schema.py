"""Table definitions and schema migrations."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

from local_memory.errors import LocalMemoryError

FTS_COLUMNS = ("title", "gist", "body", "tags")

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

RECORD_COLUMNS = (
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


#: Handlers that upgrade a database one version, keyed by the version they upgrade
#: *from*. ``MIGRATIONS[1]`` takes a v1 database to v2. A fresh database is created
#: straight at the current schema and never runs these.
MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {}


def migrate_to_schema(conn: sqlite3.Connection, target: int) -> None:
    """Bring the database to ``target``, running each handler in order.

    Each step and its version bump commit together, so an interrupted upgrade
    leaves the database at the last completed version.
    """
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > target:
        raise LocalMemoryError(f"db schema version {version} is newer than supported {target}")
    if version == 0:
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version={target}")
        return
    for current in range(version, target):
        handler = MIGRATIONS.get(current)
        if handler is None:
            raise LocalMemoryError(
                f"no migration path from schema version {current} to {current + 1}"
            )
        conn.execute("BEGIN IMMEDIATE")
        try:
            handler(conn)
            conn.execute(f"PRAGMA user_version={current + 1}")
        except BaseException:
            conn.rollback()
            raise
        conn.commit()
