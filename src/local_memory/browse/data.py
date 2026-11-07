"""Read-only data access for the viewer: listings, counts and lazy search.

Reads :class:`~local_memory.storage.Store` directly; only search goes through
the service, which is built on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from local_memory.config import Config
from local_memory.models import MemoryRecord, SearchHit, Status

if TYPE_CHECKING:
    from local_memory.storage import Store

#: Default page size. Matches the viewer's own scrolling step.
DEFAULT_PAGE = 50

#: Statuses selectable on the command line. Not just ``SEARCHABLE``: the whole
#: point of a browser is to inspect what search deliberately hides.
BROWSABLE_STATUSES: tuple[str, ...] = tuple(s.value for s in Status)


class BrowseError(Exception):
    """Viewer-level failure, rendered without a traceback."""


def list_records(
    store: Store,
    *,
    scope: str | None = None,
    kind: str | None = None,
    statuses: tuple[str, ...] | None = None,
    limit: int = DEFAULT_PAGE,
    offset: int = 0,
) -> list[MemoryRecord]:
    """Newest-first page of records matching the filters.

    A thin pass-through to :meth:`Store.list_records`; the filtering, ordering
    and pagination already live there and are tested there.
    """
    if limit < 1:
        raise BrowseError(f"limit must be >= 1, got {limit}")
    if offset < 0:
        raise BrowseError(f"offset must be >= 0, got {offset}")
    return store.list_records(
        scope=scope,
        kind=kind,
        statuses=statuses,
        limit=limit,
        offset=offset,
    )


def count_records(
    store: Store,
    *,
    scope: str | None = None,
    kind: str | None = None,
    statuses: tuple[str, ...] | None = None,
) -> int:
    """Total matching the filters, ignoring pagination.

    ``Store.count`` takes no scope/kind today, so scope and kind are applied by
    fetching the candidate ids and counting them. That is O(n) in records, which
    is irrelevant at the 10^2-10^4 scale this store targets; if it ever is not,
    extend ``Store.count`` rather than optimising here.
    """
    if scope is None and kind is None:
        return store.count(statuses=statuses)
    if statuses is not None:
        return len(store.candidate_ids(scope=scope, kind=kind, statuses=statuses))
    return len(store.candidate_ids(scope=scope, kind=kind))


class _LazySearcher:
    """Builds a :class:`MemoryService` on first search, then reuses it.

    Constructing the embedder and index is what costs ~0.2s and a model cache
    stat, so it is deferred until a query is actually typed.
    """

    def __init__(self, config: Config, store: Store) -> None:
        self._config = config
        self._store = store
        self._service: object | None = None

    @property
    def loaded(self) -> bool:
        return self._service is not None

    def _ensure(self) -> Any:
        if self._service is None:
            from local_memory.embedders.embedder import build_embedder
            from local_memory.service import MemoryService

            config = self._config
            embedder = build_embedder(
                config.default_model, config.model_cache, threads=config.threads
            )
            self._service = MemoryService(self._store, embedder, lambda: config)
        return self._service

    def search(
        self,
        query: str,
        *,
        k: int,
        scope: str | None,
        kind: str | None,
    ) -> list[SearchHit]:
        """Run a hybrid search through the real service path.

        Note that ``statuses`` is not a parameter here: search only ever sees
        ``SEARCHABLE``. A search result set therefore cannot show pending deletes
        or tombstones, which is correct behaviour and slightly surprising if you
        came from a listing that did show them.
        """
        if not query.strip():
            raise BrowseError("search query is empty")
        service = self._ensure()
        hits = service.search(query, k=k, scope=scope, kind=kind, detail="snippets")
        return cast(list[SearchHit], hits)


def _hits_to_records(store: Store, hits: list[SearchHit]) -> list[MemoryRecord]:
    """Hydrate search hits into records, preserving rank order.

    Search returns previews for context economy; the viewer wants the real rows
    so the table columns are uniform.
    """
    if not hits:
        return []
    return store.get_many([hit.id for hit in hits])
