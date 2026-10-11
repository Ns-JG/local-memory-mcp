"""Core types for the local memory service.

Holds the record schema (dataclasses), the controlled vocabularies, the
embedding-model registry, and config loading. No I/O beyond reading the
config file, so this module is cheap to import in tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

XDG_CONFIG = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
XDG_DATA = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
XDG_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

DEFAULT_CONFIG_PATH = XDG_CONFIG / "local-memory" / "config.toml"
DEFAULT_DB_PATH = XDG_DATA / "local-memory" / "memory.db"
DEFAULT_MODEL_CACHE = XDG_CACHE / "local-memory" / "models"


class Status(StrEnum):
    ACTIVE = "active"
    PENDING_DELETE = "pending_delete"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


#: Statuses that participate in search. ``pending_delete`` is excluded on
#: purpose: an agent can request a delete without the user having confirmed
#: it yet, but the record stops polluting results immediately.
SEARCHABLE = (Status.ACTIVE,)


class Scope(StrEnum):
    PERSONAL = "personal"
    WORK = "work"
    UNI = "uni"
    GENERAL = "general"


#: The ``kind`` vocabulary is deliberately open. These are seeds, not an enum;
#: agents may coin new kinds and they land in the ``kinds`` table.
SEED_KINDS: tuple[str, ...] = (
    "pattern",
    "decision",
    "project",
    "preference",
    "fact",
    "lesson",
    "correction",
)

GIST_MAX_CHARS = 200


class LocalMemoryError(Exception):
    """Base for every error this package raises."""


class NotFoundError(LocalMemoryError):
    """No record with the requested id."""


class ConflictError(LocalMemoryError):
    """A concurrent writer changed the record after it was read.

    Raised instead of silently overwriting, so the caller re-reads and retries.
    """

    def __init__(self, record_id: str) -> None:
        super().__init__(f"record {record_id} was modified concurrently; re-read it and retry")
        self.record_id = record_id


class ValidationError(LocalMemoryError):
    """A field failed validation on write."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """An embedding model and everything needed to use it correctly.

    ``query_prefix`` is the trap: BGE and E5 families expect an instruction on
    the query side, and E5 additionally expects a different one on the document
    side. Applying the query prefix to documents, or omitting it from queries,
    degrades retrieval silently. They live here so they cannot drift apart from
    the model.
    """

    name: str
    dim: int
    query_prefix: str = ""
    document_prefix: str = ""
    threads: int = 3
    note: str = ""

    def document_text(self, text: str) -> str:
        return f"{self.document_prefix}{text}" if self.document_prefix else text


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
E5_QUERY_PREFIX = "query: "
E5_DOCUMENT_PREFIX = "passage: "

#: Every entry is verified against ``fastembed.TextEmbedding.list_supported_models()``
#: by ``tests/test_models.py`` - name and dim must both match. That test exists
#: because an unverified registry entry is a trap: it resolves, it appears in
#: ``local-memory models``, and it only fails later at model-load time.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "BAAI/bge-small-en-v1.5": ModelSpec(
        name="BAAI/bge-small-en-v1.5",
        dim=384,
        query_prefix=BGE_QUERY_PREFIX,
        threads=3,
        note="default; 67MB, loads in <1s, English only",
    ),
    "minishlab/potion-base-8M": ModelSpec(
        name="minishlab/potion-base-8M",
        dim=256,
        threads=3,
        note="30MB, fastest option, English only",
    ),
    "minishlab/potion-multilingual-128M": ModelSpec(
        name="minishlab/potion-multilingual-128M",
        dim=256,
        threads=3,
        note="512MB; the cheap way to cover Polish queries",
    ),
    "jinaai/jina-embeddings-v2-base-code": ModelSpec(
        name="jinaai/jina-embeddings-v2-base-code",
        dim=768,
        threads=3,
        note="640MB, code-trained, multilingual",
    ),
    "intfloat/multilingual-e5-large": ModelSpec(
        name="intfloat/multilingual-e5-large",
        dim=1024,
        query_prefix=E5_QUERY_PREFIX,
        document_prefix=E5_DOCUMENT_PREFIX,
        threads=3,
        note="2.2GB; strongest multilingual option, slow to load",
    ),
}

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def resolve_model(name: str | None) -> ModelSpec:
    """Look up a model, falling back to the default."""
    key = name or DEFAULT_MODEL
    if key not in MODEL_REGISTRY:
        raise ValidationError(f"unknown model {key!r}; known: {', '.join(sorted(MODEL_REGISTRY))}")
    return MODEL_REGISTRY[key]


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
    #: Monotonic per-row version, bumped on every write.
    #:
    #: Deliberately *not* ``updated_at``. That field has one-second resolution,
    #: so two writes inside the same second produce an identical value and the
    #: compare-and-swap would pass when it must fail. ``rev`` is a version
    #: token; ``updated_at`` stays the human-readable wall clock used for
    #: last-writer-wins merges between devices.
    rev: int = 0

    @property
    def searchable(self) -> bool:
        return self.status in SEARCHABLE

    def compute_hash(self) -> str:
        payload = "\x1f".join([self.title, self.gist, self.body, json.dumps(self.tags)])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        """Enforce write-time invariants. Raises ValidationError."""
        if not self.title.strip():
            raise ValidationError("title must not be empty")
        if not self.body.strip():
            raise ValidationError("body must not be empty")
        if not self.kind.strip():
            raise ValidationError("kind must not be empty")
        if not self.scope.strip():
            raise ValidationError("scope must not be empty")
        if len(self.gist) > GIST_MAX_CHARS:
            raise ValidationError(f"gist must be <= {GIST_MAX_CHARS} chars, got {len(self.gist)}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError(f"confidence must be in [0,1], got {self.confidence}")


def utcnow() -> str:
    """UTC ISO-8601 with a Z suffix, second resolution."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_id() -> str:
    """UUIDv7: 48-bit millisecond timestamp, then random.

    Sorting by id orders by millisecond first, which makes ``ORDER BY id`` an
    approximate chronological index. Within a single millisecond the tail is
    random, as the UUIDv7 spec requires, so the ordering is not strictly
    monotonic - use ``created_at`` when exact order matters. Collision-free
    across devices without coordination. Python 3.13 has no ``uuid.uuid7``.

    The ``random-a`` and ``random-b`` fields are interleaved per RFC 9562
    rather than packed adjacently, so the layout is a valid v7 UUID.
    """
    ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)
    value = (ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    return str(uuid.UUID(int=value))


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


@dataclass(frozen=True, slots=True)
class ScopeRule:
    scope: str
    paths: tuple[Path, ...]
    missing: tuple[Path, ...]

    def warnings(self) -> list[str]:
        return [
            f"scope {self.scope!r} maps to {p} which does not exist - "
            f"ask NsJG to update {DEFAULT_CONFIG_PATH.name} before using it"
            for p in self.missing
        ]


@dataclass(frozen=True, slots=True)
class Config:
    db_path: Path
    model_cache: Path
    default_model: str
    threads: int
    scope_paths: dict[str, ScopeRule]
    dedup_threshold: float
    rrf_k: int
    tombstone_retention_days: int

    def scope_warnings(self) -> list[str]:
        out: list[str] = []
        for rule in self.scope_paths.values():
            out.extend(rule.warnings())
        return out

    def scope_for_path(self, path: Path) -> str:
        """Longest-prefix match of ``path`` against configured scope paths."""
        resolved = path.expanduser().resolve()
        best_scope: Scope = Scope.GENERAL
        best_len = -1
        for scope_key, rule in self.scope_paths.items():
            for candidate in rule.paths:
                root = candidate.expanduser().resolve()
                if (resolved == root or root in resolved.parents) and len(str(root)) > best_len:
                    best_scope, best_len = Scope(scope_key), len(str(root))
        return best_scope


DEFAULT_TOML = """\
# local-memory configuration
# Editing this file is the intended way to change scope mappings.

db_path = "~/.local/share/local-memory/memory.db"
model_cache = "~/.cache/local-memory/models"
default_model = "BAAI/bge-small-en-v1.5"
threads = 3
dedup_threshold = 0.92
rrf_k = 60
tombstone_retention_days = 730

[scope.personal]
paths = ["~/Desktop/priv-projects/local_memory"]

[scope.work]
paths = ["~/Desktop/VStorm"]

[scope.uni]
paths = ["~/Desktop/pwr_mgr"]
"""


def _expand(value: str) -> Path:
    return Path(value).expanduser()


def load_config(path: Path | None = None) -> Config:
    """Read config.toml, or fall back to defaults when absent.

    Never raises on a missing file - the service must start on a clean machine.
    """
    cfg_path = path or DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if cfg_path.is_file():
        raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))

    scope_paths: dict[str, ScopeRule] = {}
    for scope_name, block in (raw.get("scope") or {}).items():
        paths = tuple(_expand(p) for p in block.get("paths", []))
        scope_paths[scope_name] = ScopeRule(
            scope=scope_name,
            paths=paths,
            missing=tuple(p for p in paths if not p.exists()),
        )

    return Config(
        db_path=_expand(str(raw.get("db_path", DEFAULT_DB_PATH))),
        model_cache=_expand(str(raw.get("model_cache", DEFAULT_MODEL_CACHE))),
        default_model=str(raw.get("default_model", DEFAULT_MODEL)),
        threads=int(raw.get("threads", 3)),
        scope_paths=scope_paths,
        dedup_threshold=float(raw.get("dedup_threshold", 0.92)),
        rrf_k=int(raw.get("rrf_k", 60)),
        tombstone_retention_days=int(raw.get("tombstone_retention_days", 730)),
    )


def write_default_config(path: Path | None = None) -> Path:
    """Write the starter config.toml if it does not already exist."""
    cfg_path = path or DEFAULT_CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg_path.exists():
        cfg_path.write_text(DEFAULT_TOML, encoding="utf-8")
    return cfg_path
