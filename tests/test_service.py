"""Service: the behaviour an agent actually sees.

Retrieval, dedup-on-write, supersession, and the delete handshake.
"""

from __future__ import annotations

import dataclasses

import pytest

from local_memory.models import (
    ConflictError,
    NotFoundError,
    Status,
    ValidationError,
)
from local_memory.service import MemoryService, _derive_gist
from local_memory.store import Store


def put(service: MemoryService, title: str, body: str, **kwargs):
    return service.store_record(title=title, body=body, **kwargs)


# -- store: create ----------------------------------------------------------


def test_store_creates_record(service: MemoryService) -> None:
    record, outcome = put(
        service, "Prefer ruff over flake8", "The repo standardises on ruff for lint."
    )
    assert outcome == "created"
    assert record.status == Status.ACTIVE
    assert service.get(record.id).title == "Prefer ruff over flake8"


def test_store_derives_gist_when_omitted(service: MemoryService) -> None:
    record, _ = put(service, "t", "First sentence here. Second sentence follows.")
    assert record.gist == "First sentence here."


def test_store_respects_explicit_gist(service: MemoryService) -> None:
    record, _ = put(service, "t", "body", gist="agent-authored summary")
    assert record.gist == "agent-authored summary"


def test_store_stamps_embedding_metadata(service: MemoryService) -> None:
    record, _ = put(service, "t", "b")
    assert record.embedding_model == service.embedder.spec.name
    assert record.embedding_dim == service.embedder.spec.dim
    assert record.embedded_at is not None


def test_store_persists_vector_and_indexes_it(service: MemoryService) -> None:
    record, _ = put(service, "Vectors land in the index", "sqlite blob plus ram matrix")
    assert service.index.contains(record.id)
    assert service.store.get_vector(record.id) is not None


def test_store_registers_new_kind(service: MemoryService) -> None:
    put(service, "t", "b", kind="anti-pattern")
    assert "anti-pattern" in service.store.list_kinds()


def test_store_rejects_empty_title(service: MemoryService) -> None:
    with pytest.raises(ValidationError, match="title"):
        put(service, "   ", "body")


def test_store_rejects_empty_body(service: MemoryService) -> None:
    with pytest.raises(ValidationError, match="body"):
        put(service, "title", "   ")


# -- store: dedup -----------------------------------------------------------


def test_restating_the_same_fact_reinforces_instead_of_duplicating(
    service: MemoryService,
) -> None:
    body = "Always run the linter before committing to main branches."
    first, outcome_a = put(service, "Lint before commit", body)
    second, outcome_b = put(service, "Lint before commit", body)

    assert outcome_a == "created"
    assert outcome_b == "reinforced"
    assert second.id == first.id
    assert service.store.count() == 1
    assert second.times_confirmed == 1


def test_reinforcement_raises_confidence_and_merges_tags(
    service: MemoryService,
) -> None:
    first, _ = put(service, "Shared title", "shared body text", tags=["a"])
    second, _ = put(service, "Shared title", "shared body text", tags=["b"])
    assert first.confidence == pytest.approx(0.5)
    assert second.confidence == pytest.approx(0.6)
    assert set(second.tags) == {"a", "b"}


def test_reinforcement_confidence_is_capped_at_one(service: MemoryService) -> None:
    for _ in range(12):
        record, _ = put(service, "Repeats", "the exact same sentence every time")
    assert record.confidence == 1.0
    assert record.times_confirmed == 11


def test_different_facts_are_not_deduped(service: MemoryService) -> None:
    put(service, "First fact", "entirely unrelated content about databases")
    _, outcome = put(service, "Second fact", "something else about network stacks")
    assert outcome == "created"
    assert service.store.count() == 2


def test_dedup_can_be_disabled_for_distinct_but_similar_records(store: Store, embedder) -> None:
    from conftest import FakeEmbedder

    service = MemoryService(store, FakeEmbedder(), dedup_threshold=1.01)
    put(service, "Same", "identical body")
    _, outcome = put(service, "Same", "identical body")
    assert outcome == "created"


# -- store: supersede -------------------------------------------------------


def test_supersede_retires_predecessor(service: MemoryService) -> None:
    old, _ = put(service, "Indentation: tabs", "Use tabs for indentation.")
    new, outcome = service.supersede(
        old.id, title="Indentation: spaces", body="Use four spaces per level."
    )
    assert outcome == "superseded"
    assert new.supersedes_id == old.id
    assert service.get(old.id).status == Status.SUPERSEDED
    assert service.get(new.id).status == Status.ACTIVE


def test_superseded_record_leaves_search(service: MemoryService) -> None:
    old, _ = put(service, "Ternary operator", "Prefer explicit branching.")
    new, _ = service.supersede(old.id, title="Match statement", body="Use match for branching.")
    ids = [h.id for h in service.search("branching")]
    assert old.id not in ids
    assert new.id in ids


def test_supersede_unknown_id_raises(service: MemoryService) -> None:
    with pytest.raises(NotFoundError):
        service.supersede("no-such-id", title="t", body="b")


def test_supersede_skips_dedup(service: MemoryService) -> None:
    """An explicit replacement is never swallowed as a duplicate."""
    old, _ = put(service, "Same text", "byte for byte identical body")
    new, outcome = service.supersede(old.id, title="Same text", body="byte for byte identical body")
    assert outcome == "superseded"
    assert new.id != old.id


# -- search -----------------------------------------------------------------


@pytest.fixture
def populated(service: MemoryService) -> MemoryService:
    put(
        service,
        "Postgres connection pooling",
        "Use pgbouncer in front of postgres.",
        kind="decision",
    )
    put(service, "Python packaging", "The repo uses uv, not pip.", kind="decision", scope="work")
    put(
        service,
        "Editor preference",
        "NsJG uses neovim with lazy.nvim.",
        kind="preference",
        scope="personal",
    )
    return service


def test_search_finds_lexical_match(populated: MemoryService) -> None:
    hits = populated.search("postgres")
    assert hits
    assert hits[0].title == "Postgres connection pooling"


def test_search_hit_carries_both_channel_ranks(populated: MemoryService) -> None:
    hit = populated.search("postgres pooling")[0]
    assert hit.dense_rank is not None
    assert hit.lexical_rank is not None
    assert set(hit.found_by) == {"dense", "lexical"}


def test_search_does_not_leak_body_by_default(populated: MemoryService) -> None:
    assert populated.search("postgres")[0].body is None


def test_search_detail_titles_drops_gist(populated: MemoryService) -> None:
    hit = populated.search("postgres", detail="titles")[0]
    assert hit.gist == ""
    assert hit.body is None


def test_search_detail_full_includes_body(populated: MemoryService) -> None:
    hit = populated.search("postgres", detail="full")[0]
    assert hit.body is not None and "pgbouncer" in hit.body


def test_search_rejects_unknown_detail(populated: MemoryService) -> None:
    with pytest.raises(ValidationError, match="detail"):
        populated.search("postgres", detail="everything")


def test_search_rejects_k_below_one(populated: MemoryService) -> None:
    with pytest.raises(ValidationError, match="k"):
        populated.search("postgres", k=0)


def test_search_filters_by_scope(populated: MemoryService) -> None:
    hits = populated.search("what do you know", scope="personal")
    assert {h.scope for h in hits} == {"personal"}


def test_search_filters_by_kind(populated: MemoryService) -> None:
    hits = populated.search("what do you know", kind="decision")
    assert {h.kind for h in hits} == {"decision"}


def test_search_respects_limit(populated: MemoryService) -> None:
    assert len(populated.search("what do you know", k=1)) == 1


def test_nonsense_query_still_returns_nearest_neighbours(
    populated: MemoryService,
) -> None:
    """Dense retrieval has no similarity floor, by design.

    Cosine always has a nearest neighbour, so a nonsense query still returns
    the closest records rather than nothing. Callers judge quality via the
    per-channel ranks, not by treating a hit as a match.
    """
    hits = populated.search("zzzzzq unrelated gibberish")
    assert len(hits) == 3
    assert all(h.lexical_rank is None for h in hits)


def test_search_on_empty_store_is_empty(service: MemoryService) -> None:
    assert service.search("anything") == []


def test_search_sees_records_written_by_another_process(store: Store, embedder) -> None:
    """Two servers, one database: the reader must notice the writer."""
    from conftest import FakeEmbedder

    writer = MemoryService(store, embedder)
    reader = MemoryService(Store(store.db_path), FakeEmbedder())
    writer.store_record(title="Cross process", body="written by the other server")
    assert [h.title for h in reader.search("cross process")] == ["Cross process"]


# -- forget / confirm -------------------------------------------------------


def test_forget_is_pending_not_deleted(service: MemoryService) -> None:
    record, _ = put(service, "Stale note", "no longer relevant")
    service.forget(record.id, reason="superseded by convention")

    stored = service.get(record.id)
    assert stored.status == Status.PENDING_DELETE
    assert stored.delete_reason == "superseded by convention"
    assert stored.deleted_at is None


def test_pending_delete_disappears_from_search(service: MemoryService) -> None:
    record, _ = put(service, "Stale note", "no longer relevant")
    service.forget(record.id)
    assert service.search("stale note") == []


def test_forget_is_idempotent(service: MemoryService) -> None:
    record, _ = put(service, "Stale note", "no longer relevant")
    service.forget(record.id)
    assert service.forget(record.id).status == Status.PENDING_DELETE


def test_confirm_approve_writes_tombstone(service: MemoryService) -> None:
    record, _ = put(service, "Stale note", "no longer relevant")
    service.forget(record.id)
    assert service.confirm_delete([record.id], approve=True) == [(record.id, "deleted")]

    stored = service.get(record.id)
    assert stored.status == Status.DELETED
    assert stored.deleted_at is not None


def test_confirm_dismiss_restores(service: MemoryService) -> None:
    record, _ = put(service, "Stale note", "no longer relevant")
    service.forget(record.id)
    service.confirm_delete([record.id], approve=False)
    assert service.get(record.id).status == Status.ACTIVE
    assert service.get(record.id).delete_reason is None


def test_confirm_ignores_non_pending(service: MemoryService) -> None:
    record, _ = put(service, "Stale note", "no longer relevant")
    assert service.confirm_delete([record.id], approve=True) == [
        (record.id, "not_pending (active)")
    ]
    assert service.get(record.id).status == Status.ACTIVE


def test_confirm_reports_already_deleted(service: MemoryService) -> None:
    record, _ = put(service, "Stale note", "no longer relevant")
    service.forget(record.id)
    service.confirm_delete([record.id], approve=True)
    assert service.confirm_delete([record.id], approve=True) == [(record.id, "already_deleted")]


def test_forget_unknown_id_raises(service: MemoryService) -> None:
    with pytest.raises(NotFoundError):
        service.forget("no-such-id")


def test_restore_returns_to_active(service: MemoryService) -> None:
    record, _ = put(service, "Stale note", "no longer relevant")
    service.forget(record.id)
    assert service.restore(record.id).status == Status.ACTIVE


# -- get / stats ------------------------------------------------------------


def test_get_unknown_id_raises(service: MemoryService) -> None:
    with pytest.raises(NotFoundError):
        service.get("no-such-id")


def test_get_roundtrips_tags(service: MemoryService) -> None:
    record, _ = put(service, "Tagged", "body", tags=["a", "b"])
    assert service.get(record.id).tags == ["a", "b"]


def test_stats_reflects_the_store(populated: MemoryService) -> None:
    stats = populated.stats()
    assert stats["total"] == 3
    assert stats["searchable"] == 3
    assert stats["vectors"] == 3
    assert stats["model"] == "fake/hashed-bow"
    assert stats["dim"] == 256
    assert "decision" in stats["kinds"]


def test_stats_separates_pending_delete(populated: MemoryService) -> None:
    populated.forget(populated.store.list_records()[0].id)
    assert populated.stats()["searchable"] == 2


# -- gist derivation --------------------------------------------------------


def test_derive_gist_clips_long_first_sentence() -> None:
    gist = _derive_gist("word " * 100)
    assert len(gist) <= 200


def test_derive_gist_collapses_whitespace() -> None:
    assert _derive_gist("a\n\n  b   c.") == "a b c."


# -- concurrency ------------------------------------------------------------


def test_reinforcement_detects_concurrent_write(service: MemoryService) -> None:
    """A bump must fail loudly, not merge with somebody else's change."""
    body = "the same reusable observation"
    put(service, "Concurrent", body)
    live = service.store.list_records()[0]
    stale = live.rev

    service.store.set_status(live.id, Status.PENDING_DELETE, expected_rev=stale)

    with pytest.raises(ConflictError):
        service._reinforce(live, dataclasses.replace(live, tags=["x"]))
