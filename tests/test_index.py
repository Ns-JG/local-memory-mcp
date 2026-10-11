"""Dense index: cosine scan, masking, epoch invalidation, RRF fusion."""

from __future__ import annotations

import numpy as np
import pytest

from local_memory.index import DenseIndex, prefetch_count, rrf_fuse
from local_memory.models import LocalMemoryError, MemoryRecord, new_id
from local_memory.store import Store


def rec(**overrides) -> MemoryRecord:
    defaults = {
        "id": new_id(),
        "kind": "pattern",
        "scope": "work",
        "title": "t",
        "gist": "g",
        "body": "b",
    }
    return MemoryRecord(**(defaults | overrides))


# -- search -----------------------------------------------------------------


def test_returns_nearest_first(store: Store) -> None:
    index = DenseIndex(store, 3)
    a, b, c = rec(), rec(), rec()
    index.add(a.id, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    index.add(b.id, np.array([0.9, 0.1, 0.0], dtype=np.float32))
    index.add(c.id, np.array([0.0, 0.0, 1.0], dtype=np.float32))

    hits = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), limit=3)
    assert [rid for rid, _ in hits] == [a.id, b.id, c.id]
    assert hits[0][1] == pytest.approx(1.0)
    assert hits[2][1] == pytest.approx(0.0)


def test_respects_limit(store: Store) -> None:
    index = DenseIndex(store, 2)
    for i in range(5):
        index.add(new_id(), np.array([1.0, float(i)], dtype=np.float32))
    assert len(index.search(np.array([1.0, 0.0], dtype=np.float32), limit=2)) == 2


def test_allowed_ids_filters(store: Store) -> None:
    index = DenseIndex(store, 2)
    keep, drop = rec(), rec()
    index.add(keep.id, np.array([1.0, 0.0], dtype=np.float32))
    index.add(drop.id, np.array([1.0, 0.0], dtype=np.float32))

    hits = index.search(np.array([1.0, 0.0], dtype=np.float32), allowed_ids={keep.id})
    assert [rid for rid, _ in hits] == [keep.id]


def test_allowed_ids_with_no_overlap_returns_empty(store: Store) -> None:
    index = DenseIndex(store, 2)
    index.add(new_id(), np.array([1.0, 0.0], dtype=np.float32))
    assert index.search(np.array([1.0, 0.0], dtype=np.float32), allowed_ids={"nope"}) == []


def test_empty_index_returns_empty(store: Store) -> None:
    assert DenseIndex(store, 4).search(np.zeros(4, dtype=np.float32)) == []


def test_query_dim_mismatch_raises(store: Store) -> None:
    index = DenseIndex(store, 4)
    index.add(new_id(), np.zeros(4, dtype=np.float32))
    with pytest.raises(LocalMemoryError, match="reindex"):
        index.search(np.zeros(8, dtype=np.float32))


def test_adding_wrong_dim_raises(store: Store) -> None:
    with pytest.raises(LocalMemoryError, match="expects"):
        DenseIndex(store, 4).add(new_id(), np.zeros(7, dtype=np.float32))


# -- mutation ---------------------------------------------------------------


def test_add_replaces_existing(store: Store) -> None:
    index = DenseIndex(store, 2)
    rid = new_id()
    index.add(rid, np.array([1.0, 0.0], dtype=np.float32))
    index.add(rid, np.array([0.0, 1.0], dtype=np.float32))
    assert len(index) == 1
    top = index.search(np.array([0.0, 1.0], dtype=np.float32))[0]
    assert top[0] == rid and top[1] == pytest.approx(1.0)


def test_remove_reindexes_positions(store: Store) -> None:
    index = DenseIndex(store, 2)
    keep, drop = new_id(), new_id()
    index.add(keep, np.array([1.0, 0.0], dtype=np.float32))
    index.add(drop, np.array([0.0, 1.0], dtype=np.float32))
    index.remove(drop)
    assert len(index) == 1
    assert index.ids == [keep]
    assert index.contains(keep) and not index.contains(drop)


def test_remove_unknown_is_noop(store: Store) -> None:
    index = DenseIndex(store, 2)
    index.add(new_id(), np.zeros(2, dtype=np.float32))
    index.remove("absent")
    assert len(index) == 1


# -- cross-process invalidation --------------------------------------------


def test_refresh_if_stale_detects_external_write(store: Store) -> None:
    index = DenseIndex(store, 2)
    rid = store.insert(rec()).id
    index.add(rid, np.array([1.0, 0.0], dtype=np.float32))

    store.put_vector(rid, np.array([0.0, 1.0], dtype=np.float32))
    assert index.refresh_if_stale() is True

    top = index.search(np.array([0.0, 1.0], dtype=np.float32))[0]
    assert top[0] == rid and top[1] == pytest.approx(1.0)


def test_refresh_if_stale_is_quiet_when_unchanged(store: Store) -> None:
    index = DenseIndex(store, 2)
    index.add(new_id(), np.zeros(2, dtype=np.float32))
    assert index.refresh_if_stale() is False


def test_stale_dim_on_load_raises(store: Store) -> None:
    rec_id = store.insert(rec()).id
    store.put_vector(rec_id, np.zeros(16, dtype=np.float32))
    with pytest.raises(LocalMemoryError, match="reindex"):
        DenseIndex(store, 4)


# -- RRF --------------------------------------------------------------------


def test_rrf_promotes_records_found_by_both_channels() -> None:
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    lexical = [("c", 12.0), ("a", 9.0), ("d", 4.0)]
    fused = rrf_fuse(dense, lexical, k=60)
    order = [rid for rid, *_ in fused]
    assert order[0] == "a"  # rank 0 in dense, rank 1 in lexical
    assert order[1] == "c"  # rank 2 in dense, rank 0 in lexical
    assert set(order) == {"a", "b", "c", "d"}


def test_rrf_uses_ranks_not_raw_scores() -> None:
    """A BM25 score of 900 must not outweigh a rank-0 dense hit."""
    fused = rrf_fuse([("a", 0.01)], [("b", 900.0), ("c", 800.0)], k=60)
    assert [rid for rid, *_ in fused] == ["a", "b", "c"]


def test_rrf_math_matches_definition() -> None:
    """Rank 0 in both channels is 2/61, not 1/61 + 1/62."""
    fused = rrf_fuse([("a", 1.0), ("b", 1.0)], [("a", 1.0)], k=60)
    scores = {rid: score for rid, score, *_ in fused}
    assert scores["a"] == pytest.approx(2 / 61)
    assert scores["b"] == pytest.approx(1 / 62)


def test_rrf_reports_per_channel_ranks() -> None:
    fused = rrf_fuse([("a", 1.0), ("b", 1.0)], [("b", 1.0)], k=60)
    ranks = {rid: (dr, lr) for rid, _, dr, lr in fused}
    assert ranks["a"] == (0, None)
    assert ranks["b"] == (1, 0)


def test_rrf_respects_limit() -> None:
    dense = [(f"d{i}", 1.0) for i in range(10)]
    assert len(rrf_fuse(dense, [], k=60, limit=3)) == 3


def test_rrf_with_no_hits_is_empty() -> None:
    assert rrf_fuse([], [], k=60) == []


def test_rrf_rejects_k_below_one() -> None:
    with pytest.raises(ValueError, match="k must be"):
        rrf_fuse([("a", 1.0)], [], k=0)


def test_rrf_default_k_is_60() -> None:
    fused = rrf_fuse([("a", 1.0)], [], k=60)
    assert fused[0][1] == pytest.approx(1 / 61)


# -- prefetch sizing --------------------------------------------------------


def test_prefetch_is_wider_than_k() -> None:
    assert prefetch_count(3) >= 3
    assert prefetch_count(50) >= 200
    assert prefetch_count(1) >= 20
