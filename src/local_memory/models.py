"""Core types for the local memory service: the record schema and search results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum

from local_memory.errors import ValidationError


class Status(StrEnum):
    ACTIVE = "active"
    PENDING_DELETE = "pending_delete"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


# Only active records are searched.
# A pending_delete record drops out immediately,
# before the user has confirmed the deletion.
SEARCHABLE = (Status.ACTIVE,)


@dataclass(slots=True)
class MemoryRecord:
    id: str
    kind: str
    scope: str
    title: str
    gist: str
    body: str
    tags: list[str] = field(default_factory=list)
    status: str = Status.ACTIVE
    supersedes_id: str | None = None
    delete_reason: str | None = None
    confidence: float = 0.5
    source: str = "agent_inferred"
    times_confirmed: int = 0
    content_hash: str = ""
    embedding_model: str | None = None
    embedding_dim: int | None = None
    embedded_at: str | None = None
    created_at: str = ""
    updated_at: str = ""
    deleted_at: str | None = None
    origin_device: str | None = None
    # Monotonic per-row version, bumped on every write.
    #
    # Deliberately *not* ``updated_at``. That field has one-second resolution,
    # so two writes inside the same second produce an identical value and the
    # compare-and-swap would pass when it must fail. ``rev`` is a version
    # token; ``updated_at`` stays the human-readable wall clock used for
    # last-writer-wins merges between devices.
    rev: int = 0

    @property
    def searchable(self) -> bool:
        return self.status in SEARCHABLE

    def compute_hash(self) -> str:
        payload = "\x1f".join([self.title, self.gist, self.body, json.dumps(self.tags)])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self, *, gist_max_chars: int | None = None) -> None:
        """Enforce write-time invariants. Raises ValidationError.

        The gist limit is configurable, so it is only checked when the caller passes it.
        """
        if not self.title.strip():
            raise ValidationError("title must not be empty")
        if not self.body.strip():
            raise ValidationError("body must not be empty")
        if not self.kind.strip():
            raise ValidationError("kind must not be empty")
        if not self.scope.strip():
            raise ValidationError("scope must not be empty")
        if gist_max_chars is not None and len(self.gist) > gist_max_chars:
            raise ValidationError(f"gist must be <= {gist_max_chars} chars, got {len(self.gist)}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError(f"confidence must be in [0,1], got {self.confidence}")


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One search result.

    ``dense_rank`` / ``lexical_rank`` are 0-based, ``None`` when that channel
    did not return the record. RRF scores are not comparable across queries, so
    the per-channel ranks are what let a caller judge a hit.
    """

    id: str
    title: str
    gist: str
    kind: str
    scope: str
    tags: list[str]
    fused_score: float
    dense_rank: int | None
    lexical_rank: int | None
    body: str | None = None

    @property
    def found_by(self) -> list[str]:
        channels = []
        if self.dense_rank is not None:
            channels.append("dense")
        if self.lexical_rank is not None:
            channels.append("lexical")
        return channels
