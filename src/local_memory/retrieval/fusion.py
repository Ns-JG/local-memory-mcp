"""Reciprocal Rank Fusion of the dense and lexical channels."""

from __future__ import annotations

DEFAULT_PREFETCH_K = 20
MIN_PREFETCH_MULTIPLIER = 4


def prefetch_count(k: int) -> int:
    return max(MIN_PREFETCH_MULTIPLIER * k, DEFAULT_PREFETCH_K)


def rrf_fuse(
    dense_hits: list[tuple[str, float]],
    lexical_hits: list[tuple[str, float]],
    *,
    k: int = 60,
    limit: int = 10,
) -> list[tuple[str, float, int | None, int | None]]:
    """Reciprocal Rank Fusion. Returns ``(id, score, dense_rank, lexical_rank)``.

    ``k=60`` is the value from the original Cormack et al. paper. RRF consumes
    only rank, never raw score, which is exactly why it works across a cosine
    score and a BM25 score that share no scale. The raw scores are discarded -
    this is also why the per-channel ranks are returned, so callers can report
    something an LLM can reason about.
    """
    if k < 1:
        raise ValueError("rrf k must be >= 1")

    dense_rank = {rid: rank for rank, (rid, _) in enumerate(dense_hits)}
    lexical_rank = {rid: rank for rank, (rid, _) in enumerate(lexical_hits)}

    scores: dict[str, float] = {}
    for ranks in (dense_rank, lexical_rank):
        for record_id, rank in ranks.items():
            scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (k + rank + 1)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        (record_id, score, dense_rank.get(record_id), lexical_rank.get(record_id))
        for record_id, score in ordered[:limit]
    ]
