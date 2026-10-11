"""The memory service: search, get, store, forget, confirm.

Pure with respect to I/O policy - it talks to a :class:`Store` and an
:class:`Embedder` and nothing else. No network calls, no LLM, no environment
reads. That is what makes it testable with a fake embedder in milliseconds and
what keeps the stdio and HTTP transports interchangeable.

Two behaviours are worth reading before using it:

* **Dedup on write.** Before inserting, the incoming record is embedded and
  compared against the live index. A cosine at or above ``dedup_threshold`` is
  treated as a re-observation of an existing record, so the existing one is
  reinforced rather than duplicated. Without this, an agent that re-states the
  same pattern five times leaves five rows competing in every later search.
* **Deletes are requests, not actions.** :meth:`forget` only moves a record to
  ``pending_delete``. Nothing reaches ``deleted`` until :meth:`confirm_delete`
  runs with approval.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .embedder import Embedder
from .index import DenseIndex, prefetch_count, rrf_fuse
from .models import (
    SEARCHABLE,
    MemoryRecord,
    SearchHit,
    Status,
    ValidationError,
    new_id,
    utcnow,
)
from .store import Store

DETAIL_TIERS = ("titles", "snippets", "full")
DEFAULT_K = 5
REINFORCE_CONFIDENCE_STEP = 0.1


def embed_text(title: str, body: str) -> str:
    """What actually gets embedded for a document.

    Title first: it is the compressed identity of the record, and BM25 already
    weights it 10x. The gist is omitted because it is derived from the body and
    would only skew the vector.
    """
    return f"{title}\n{body}"


class MemoryService:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        *,
        dedup_threshold: float = 0.92,
        rrf_k: int = 60,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.dedup_threshold = dedup_threshold
        self.rrf_k = rrf_k
        self.index = DenseIndex(store, embedder.spec.dim)

    # -- read --------------------------------------------------------------

    def get(self, record_id: str) -> MemoryRecord:
        return self.store.get(record_id)

    def search(
        self,
        query: str,
        *,
        k: int = DEFAULT_K,
        scope: str | None = None,
        kind: str | None = None,
        detail: str = "snippets",
    ) -> list[SearchHit]:
        """Filter, retrieve both channels, fuse, hydrate.

        Filters run before retrieval: they are cheaper than similarity and more
        precise for structured questions like "what do you know about my
        preferences".
        """
        if detail not in DETAIL_TIERS:
            raise ValidationError(f"detail must be one of {DETAIL_TIERS}, got {detail!r}")
        if k < 1:
            raise ValidationError(f"k must be >= 1, got {k}")

        allowed = set(self.store.candidate_ids(scope=scope, kind=kind, statuses=SEARCHABLE))
        if not allowed:
            return []

        prefetch = prefetch_count(k)
        query_vector = self.embedder.embed_query(query)

        dense = self.index.search(query_vector, allowed_ids=allowed, limit=prefetch)
        lexical = self.store.lexical_search(
            query, scope=scope, kind=kind, statuses=SEARCHABLE, limit=prefetch
        )
        fused = rrf_fuse(dense, lexical, k=self.rrf_k, limit=k)
        if not fused:
            return []

        records = {r.id: r for r in self.store.get_many([rid for rid, *_ in fused])}
        hits: list[SearchHit] = []
        for record_id, score, dense_rank, lexical_rank in fused:
            record = records.get(record_id)
            if record is None:
                continue
            hits.append(
                SearchHit(
                    id=record.id,
                    title=record.title,
                    gist="" if detail == "titles" else record.gist,
                    kind=record.kind,
                    scope=record.scope,
                    tags=record.tags,
                    fused_score=score,
                    dense_rank=dense_rank,
                    lexical_rank=lexical_rank,
                    body=record.body if detail == "full" else None,
                )
            )
        return hits

    def stats(self) -> dict[str, object]:
        by_status: dict[str, int] = {}
        for status in Status:
            by_status[str(status)] = self.store.count(statuses=[str(status)])
        return {
            "total": self.store.count(),
            "searchable": self.store.count(statuses=list(SEARCHABLE)),
            "by_status": by_status,
            "vectors": len(self.index),
            "kinds": self.store.list_kinds(),
            "model": self.embedder.spec.name,
            "dim": self.embedder.spec.dim,
            "epoch": self.store.epoch(),
        }

    # -- write -------------------------------------------------------------

    def store_record(
        self,
        *,
        title: str,
        body: str,
        gist: str = "",
        kind: str = "pattern",
        scope: str = "general",
        tags: Sequence[str] | None = None,
        supersedes_id: str | None = None,
        confidence: float = 0.5,
        source: str = "agent_inferred",
    ) -> tuple[MemoryRecord, str]:
        """Create or reinforce. Returns ``(record, outcome)``.

        ``outcome`` is ``"created"``, ``"reinforced"``, or ``"superseded"``.
        """
        if not gist:
            gist = _derive_gist(body)

        candidate = MemoryRecord(
            id=new_id(),
            kind=kind,
            scope=scope,
            title=title,
            gist=gist,
            body=body,
            tags=list(tags or []),
            supersedes_id=supersedes_id,
            confidence=confidence,
            source=source,
        )
        candidate.validate()

        vector = self.embedder.embed_documents([embed_text(candidate.title, candidate.body)])[0]

        if supersedes_id is None:
            duplicate = self._find_duplicate(vector)
            if duplicate is not None:
                return self._reinforce(duplicate, candidate), "reinforced"

        candidate.embedding_model = self.embedder.spec.name
        candidate.embedding_dim = self.embedder.spec.dim
        candidate.embedded_at = utcnow()

        if supersedes_id is None:
            stored = self.store.insert(candidate)
            outcome = "created"
        else:
            predecessor = self.store.get(supersedes_id)
            stored = self.store.insert_and_supersede(candidate, supersedes_id, predecessor.rev)
            outcome = "superseded"

        # Durable first, cached second. put_vector bumps the epoch, so the next
        # refresh reloads from SQLite - the in-memory matrix is a cache of the
        # vectors table, never the source of truth.
        self.store.put_vector(stored.id, vector)
        self.index.add(stored.id, vector)
        return stored, outcome

    def _find_duplicate(self, vector: np.ndarray, limit: int = 5) -> MemoryRecord | None:
        """Nearest live record above the dedup threshold, if any."""
        self.index.refresh_if_stale()
        allowed = set(self.store.candidate_ids(statuses=list(SEARCHABLE)))
        for record_id, score in self.index.search(vector, allowed_ids=allowed, limit=limit):
            if score >= self.dedup_threshold:
                return self.store.get(record_id)
        return None

    def _reinforce(self, existing: MemoryRecord, candidate: MemoryRecord) -> MemoryRecord:
        """Record a re-observation instead of creating a near-duplicate row."""
        existing.confidence = min(1.0, existing.confidence + REINFORCE_CONFIDENCE_STEP)
        for tag in candidate.tags:
            if tag not in existing.tags:
                existing.tags.append(tag)
        return self.store.update(existing, existing.rev, bump_confirmed=True)

    def forget(self, record_id: str, reason: str = "") -> MemoryRecord:
        """Request a delete. Moves to ``pending_delete``; nothing is destroyed.

        Pending records are excluded from search immediately, so the caller gets
        the effect they asked for, but the row stays fully recoverable until the
        user confirms.
        """
        record = self.store.get(record_id)
        if record.status == Status.DELETED:
            raise ValidationError(f"record {record_id} is already deleted")
        if record.status == Status.PENDING_DELETE:
            return record
        return self.store.set_status(
            record_id,
            Status.PENDING_DELETE,
            expected_rev=record.rev,
            reason=reason or None,
        )

    def confirm_delete(self, record_ids: Sequence[str], approve: bool) -> list[tuple[str, str]]:
        """Approve or dismiss pending deletes.

        Approving writes the tombstone (which is what makes the delete
        propagate through ``merge``). Dismissing restores the record.
        """
        target = Status.DELETED if approve else Status.ACTIVE
        results: list[tuple[str, str]] = []
        for record_id in record_ids:
            record = self.store.get(record_id)
            if record.status == Status.DELETED:
                results.append((record_id, "already_deleted"))
                continue
            if record.status != Status.PENDING_DELETE:
                results.append((record_id, f"not_pending ({record.status})"))
                continue
            reason = record.delete_reason if approve else None
            self.store.set_status(
                record_id,
                target,
                expected_rev=record.rev,
                reason=reason,
            )
            results.append((record_id, "deleted" if approve else "restored"))
        return results

    def restore(self, record_id: str) -> MemoryRecord:
        record = self.store.get(record_id)
        if record.status == Status.ACTIVE:
            return record
        return self.store.set_status(record_id, Status.ACTIVE, reason=None)

    def supersede(self, record_id: str, **kwargs: object) -> tuple[MemoryRecord, str]:
        """Replace a record with a new one, retiring the old in one transaction."""
        self.store.get(record_id)  # raises NotFoundError
        return self.store_record(supersedes_id=record_id, **kwargs)  # type: ignore[arg-type]


def _derive_gist(body: str, limit: int = 200) -> str:
    """First sentence, clipped to the preview budget.

    Deterministic on purpose: a server-side LLM call would make writes
    non-reproducible and would put an API key in the request path. The calling
    agent can always pass an explicit, better gist.
    """
    text = " ".join(body.split())
    for terminator in (". ", "? ", "! ", "\n"):
        index = text.find(terminator)
        if 0 < index <= limit:
            return text[: index + 1].strip()
    return text[:limit].strip()
