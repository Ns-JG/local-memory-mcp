import sqlite3

import pytest

from local_memory.errors import LocalMemoryError
from local_memory.storage import MIGRATIONS, migrate_to_schema


@pytest.fixture
def conn() -> sqlite3.Connection:
    return sqlite3.connect(":memory:", isolation_level=None)


def version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def test_fresh_database_is_created_at_the_target_version(conn) -> None:
    migrate_to_schema(conn, 3)
    assert version(conn) == 3
    assert conn.execute("SELECT count(*) FROM records").fetchone()[0] == 0


def test_database_newer_than_the_code_is_refused(conn) -> None:
    migrate_to_schema(conn, 2)
    with pytest.raises(LocalMemoryError, match="newer than supported"):
        migrate_to_schema(conn, 1)


def test_missing_handler_is_a_clear_error(conn) -> None:
    migrate_to_schema(conn, 1)
    with pytest.raises(LocalMemoryError, match="no migration path from schema version 1 to 2"):
        migrate_to_schema(conn, 2)
    assert version(conn) == 1


def test_handlers_run_in_order_and_bump_the_version(conn, monkeypatch) -> None:
    ran: list[int] = []
    monkeypatch.setitem(MIGRATIONS, 1, lambda c: ran.append(1))
    monkeypatch.setitem(MIGRATIONS, 2, lambda c: ran.append(2))
    migrate_to_schema(conn, 1)
    migrate_to_schema(conn, 3)
    assert ran == [1, 2]
    assert version(conn) == 3


def test_failed_handler_leaves_the_previous_version(conn, monkeypatch) -> None:
    def boom(c: sqlite3.Connection) -> None:
        c.execute("CREATE TABLE half_done (x)")
        raise RuntimeError("boom")

    monkeypatch.setitem(MIGRATIONS, 1, boom)
    migrate_to_schema(conn, 1)
    with pytest.raises(RuntimeError):
        migrate_to_schema(conn, 2)
    assert version(conn) == 1
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='half_done'").fetchone() is None
