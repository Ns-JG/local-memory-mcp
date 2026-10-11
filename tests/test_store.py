"""Store: schema, FTS5 trigger sync, CAS writes, epoch, vectors."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from local_memory.models import (
    ConflictError,
    MemoryRecord,
    NotFoundError,
    Status,
    new_id,
    utcnow,
)
from local_memory.store import Store, sanitize_fts_query


def record(**overrides) -> MemoryRecord:
    defaults = {
        "id": new_id(),
        "kind": "pattern",
        "scope": "work",
        "title": "Prefer composition over inheritance",
        "gist": "Favour composition.",
        "body": "When two classes share behaviour, extract a collaborator "
        "rather than deepening the base class.",
        "tags": ["python", "design"],
    }
    return MemoryRecord(**(defaults | overrides))


# -- schema -----------------------------------------------------------------


def test_schema_version_is_current(store: Store) -> None:
    version = store._conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == 1


def test_wal_is_enabled(store: Store) -> None:
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_reopen_preserves_data(tmp_path) -> None:
    path = tmp_path / "memory.db"
    with Store(path) as first:
        first.insert(record(title="Persisted record about uv"))
    with Store(path) as second:
        assert second.count() == 1
        assert second.all_records()[0].title == "Persisted record about uv"


# -- FTS5 trigger sync ------------------------------------------------------


def test_fts_triggers_index_on_insert(store: Store) -> None:
    rec = store.insert(record(body="Use uv for dependency management in Python"))
    hits = store.lexical_search("uv")
    assert rec.id in [rid for rid, _ in hits]


def test_fts_triggers_reindex_on_update(store: Store) -> None:
    rec = store.insert(record(body="original body text about asyncio"))
    assert store.lexical_search("asyncio")

    store.update(
        dataclasses.replace(rec, body="replacement prose regarding httpx"),
        expected_rev=rec.rev,
    )
    assert [rid for rid, _ in store.lexical_search("asyncio")] == []
    assert rec.id in [rid for rid, _ in store.lexical_search("httpx")]


def test_fts_bm25_ranks_title_above_body(store: Store) -> None:
    titled = store.insert(record(title="chroma", body="unrelated filler words"))
    buried = store.insert(record(title="unrelated filler words", body="mentions chroma"))
    hits = [rid for rid, _ in store.lexical_search("chroma")]
    assert hits[0] == titled.id
    assert buried.id in hits


def test_tombstone_stays_lexically_reachable(store: Store) -> None:
    """A deleted record is excluded by default, not erased from the FTS index.

    Tombstones must survive to be exported and merged; dropping them at delete
    time would let a second device resurrect the record.
    """
    rec = store.insert(record(body="temporary kubernetes rollout notes"))
    store.set_status(rec.id, Status.DELETED)
    assert store.lexical_search("kubernetes") == []
    assert [rid for rid, _ in store.lexical_search("kubernetes", statuses=None)] == [rec.id]


# -- query safety -----------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ['"unterminated', "NEAR(a b)", "*", "a OR b", "AND", "'; DROP TABLE records;--"],
)
def test_sanitize_emits_only_quoted_phrase_tokens(raw: str) -> None:
    cleaned = sanitize_fts_query(raw)
    for token in cleaned.split():
        assert token.startswith('"') and token.endswith('"')


def test_lexical_search_survives_hostile_input(store: Store) -> None:
    store.insert(record(body="a normal searchable body"))
    for hostile in ['"', '""', "*", "NEAR(", "a AND", "^x"]:
        store.lexical_search(hostile)  # must not raise sqlite3.OperationalError


def test_lexical_search_on_empty_tokens_returns_nothing(store: Store) -> None:
    store.insert(record(body="something"))
    assert store.lexical_search("!!! ???") == []


def test_lexical_and_pass_prefers_full_term_matches(store: Store) -> None:
    both = store.insert(record(body="postgres connection pooling notes")).id
    store.insert(record(body="postgres notes only"))
    hits = store.lexical_search("postgres connection", limit=20)
    assert hits[0][0] == both


def test_lexical_tops_up_with_or_when_and_is_too_narrow(store: Store) -> None:
    """AND alone would drop the lexical channel for an ordinary question."""
    pg = store.insert(record(body="Put pgbouncer in front of postgres in production")).id
    vim = store.insert(record(body="neovim plugins, plus some postgres trivia")).id

    hits = store.lexical_search("postgres connection", limit=20)
    assert {rid for rid, _ in hits} == {pg, vim}


def test_lexical_skips_or_pass_when_and_satisfies_the_limit(store: Store) -> None:
    rec_id = store.insert(record(body="postgres connection pooling notes")).id
    store.insert(record(body="postgres notes only"))
    hits = store.lexical_search("postgres connection", limit=1)
    assert [rid for rid, _ in hits] == [rec_id]


def test_lexical_and_results_take_precedence_over_or(store: Store) -> None:
    both = store.insert(record(title="postgresql connection guide", body="postgresql connection"))
    store.insert(record(body="unrelated postgres notes"))
    hits = store.lexical_search("postgres connection", limit=20)
    assert hits[0][0] == both.id


def test_lexical_single_token_skips_the_or_pass(store: Store) -> None:
    rec_id = store.insert(record(body="kubernetes rollout")).id
    assert [r for r, _ in store.lexical_search("kubernetes")] == [rec_id]


# -- compare-and-swap -------------------------------------------------------


def test_update_succeeds_with_current_version(store: Store) -> None:
    rec = store.insert(record())
    rec.body = "amended body"
    store.update(rec, expected_rev=rec.rev)
    assert store.get(rec.id).body == "amended body"


def test_update_with_stale_version_raises_conflict(store: Store) -> None:
    rec = store.insert(record())
    stale = rec.rev

    rec.body = "first writer wins"
    store.update(rec, expected_rev=stale)

    other = store.get(rec.id)
    other.body = "second writer must lose"
    with pytest.raises(ConflictError) as excinfo:
        store.update(other, expected_rev=stale)
    assert excinfo.value.record_id == rec.id
    assert store.get(rec.id).body == "first writer wins"


def test_rev_increments_on_every_write(store: Store) -> None:
    rec = store.insert(record())
    assert rec.rev == 1
    rec = store.update(rec, expected_rev=rec.rev)
    assert rec.rev == 2
    rec = store.set_status(rec.id, Status.PENDING_DELETE, expected_rev=rec.rev)
    assert rec.rev == 3


def test_cas_detects_conflict_within_the_same_second(store: Store) -> None:
    """The reason CAS keys on ``rev`` and not ``updated_at``.

    ``utcnow()`` has one-second resolution, so a read-then-write inside one
    second carries an unchanged timestamp. Keying the swap on that timestamp
    would let the stale writer through.
    """
    first = store.insert(record())
    stale_read = store.get(first.id)
    assert stale_read.updated_at == first.updated_at

    store.update(first, expected_rev=first.rev)
    with pytest.raises(ConflictError):
        store.update(stale_read, expected_rev=stale_read.rev)


def test_update_of_missing_record_raises_not_found(store: Store) -> None:
    with pytest.raises(NotFoundError):
        store.update(record(id=new_id()), expected_rev=1)


def test_set_status_enforces_cas(store: Store) -> None:
    rec = store.insert(record())
    with pytest.raises(ConflictError):
        store.set_status(rec.id, Status.PENDING_DELETE, expected_rev=999)


def test_delete_sets_tombstone_timestamp(store: Store) -> None:
    rec = store.insert(record())
    store.set_status(rec.id, Status.DELETED)
    assert store.get(rec.id).deleted_at is not None
    assert store.get(rec.id).status == Status.DELETED


# -- insert_and_supersede ---------------------------------------------------


def test_supersede_flips_both_rows(store: Store) -> None:
    old = store.insert(record(body="prefers tabs for indentation"))
    new = record(title="Prefer spaces", body="four spaces per indent level")
    stored = store.insert_and_supersede(new, old.id, old.rev)
    assert store.get(old.id).status == Status.SUPERSEDED
    assert store.get(stored.id).status == Status.ACTIVE


def test_supersede_is_atomic_on_conflict(store: Store) -> None:
    old = store.insert(record(body="prefers tabs for indentation"))
    stale = old.rev
    store.set_status(old.id, Status.PENDING_DELETE, expected_rev=stale)

    with pytest.raises(ConflictError):
        store.insert_and_supersede(record(body="spaces"), old.id, stale)
    # the successor must not exist - the transaction rolled back
    assert store.count() == 1


# -- epoch ------------------------------------------------------------------


def test_epoch_increments_only_on_write(store: Store) -> None:
    start = store.epoch()
    store.insert(record())
    after_write = store.epoch()
    assert after_write == start + 1

    store.count()
    store.lexical_search("anything")
    assert store.epoch() == after_write


def test_meta_roundtrip(store: Store) -> None:
    store.set_meta("model", "fake/hashed-bow")
    assert store.get_meta("model") == "fake/hashed-bow"
    assert store.get_meta("absent", "fallback") == "fallback"


# -- vectors ----------------------------------------------------------------


def test_vector_roundtrip(store: Store) -> None:
    rec = store.insert(record())
    vector = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    store.put_vector(rec.id, vector)
    assert np.allclose(store.get_vector(rec.id), vector)


def test_put_vector_requires_an_existing_record(store: Store) -> None:
    """vectors.id is a foreign key; a vector for a phantom id is a bug."""
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        store.put_vector(new_id(), np.ones(4, dtype=np.float32))


def test_put_vector_upserts(store: Store) -> None:
    rec = store.insert(record())
    store.put_vector(rec.id, np.ones(4, dtype=np.float32))
    store.put_vector(rec.id, np.zeros(4, dtype=np.float32))
    assert np.allclose(store.get_vector(rec.id), np.zeros(4))


def test_load_vectors_empty_shape(store: Store) -> None:
    ids, matrix = store.load_vectors(8)
    assert ids == []
    assert matrix.shape == (0, 8)


def test_load_vectors_stacked_in_insertion_order(store: Store) -> None:
    a, b = record(), record()
    store.insert(a)
    store.insert(b)
    store.put_vector(a.id, np.array([1.0, 0.0], dtype=np.float32))
    store.put_vector(b.id, np.array([0.0, 1.0], dtype=np.float32))
    ids, matrix = store.load_vectors(2)
    assert ids == [a.id, b.id]
    assert matrix.shape == (2, 2)


def test_stale_records_detects_missing_and_wrong_model(store: Store) -> None:
    fresh = store.insert(record())
    store.set_embedding_info(fresh.id, "BAAI/bge-small-en-v1.5", 384)
    old = store.insert(record())

    # never embedded, or embedded with the wrong model
    assert set(store.stale_records("BAAI/bge-small-en-v1.5")) == {fresh.id, old.id}

    store.put_vector(fresh.id, np.zeros(384, dtype=np.float32))
    assert store.stale_records("BAAI/bge-small-en-v1.5") == [old.id]
    assert set(store.stale_records("BAAI/bge-m3")) == {fresh.id, old.id}


# -- filtering --------------------------------------------------------------


def test_candidate_ids_respect_filters(store: Store) -> None:
    work = store.insert(record(scope="work", kind="pattern"))
    personal = store.insert(record(scope="personal", kind="preference"))
    assert set(store.candidate_ids(scope="work")) == {work.id}
    assert set(store.candidate_ids(kind="preference")) == {personal.id}
    assert set(store.candidate_ids()) == {work.id, personal.id}


def test_pending_delete_is_not_candidate(store: Store) -> None:
    rec = store.insert(record())
    store.set_status(rec.id, Status.PENDING_DELETE)
    assert store.candidate_ids() == []


def test_list_records_paginates(store: Store) -> None:
    made = [store.insert(record(title=f"item {i}")) for i in range(5)]
    page = store.list_records(limit=2, offset=0)
    assert len(page) == 2
    assert len(store.list_records(limit=2, offset=4)) == 1
    assert {r.id for r in page} <= {r.id for r in made}


def test_kinds_table_seeded_on_insert(store: Store) -> None:
    store.insert(record(kind="decision"))
    store.insert(record(kind="lesson"))
    assert "decision" in store.list_kinds()
    store.add_kind("custom-kind", note="added by cli")
    assert "custom-kind" in store.list_kinds()


def test_validation_rejects_empty_title() -> None:
    with pytest.raises(Exception, match="title"):
        record(title="   ").validate()


def test_validation_rejects_oversized_gist() -> None:
    with pytest.raises(Exception, match="gist"):
        record(gist="x" * 500).validate()


def test_content_hash_tracks_body(store: Store) -> None:
    rec = store.insert(record())
    original = rec.content_hash
    rec.body = "completely different text"
    store.update(rec, expected_rev=rec.rev)
    assert store.get(rec.id).content_hash != original


def test_utcnow_is_iso_z() -> None:
    stamp = utcnow()
    assert stamp.endswith("Z")
    assert len(stamp) == 20
