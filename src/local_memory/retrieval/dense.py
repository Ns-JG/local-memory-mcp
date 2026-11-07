"""Dense index: a brute-force cosine scan over a float32 matrix held in memory.

At the scale this system targets (10^2 to 10^4 records) that beats an HNSW
index: no build step, no rebuild when the model changes, no index corruption,
and a 15 MB matrix reloads in about 10 ms.

Cross-process staleness is handled by ``store_epoch``. Every commit bumps it,
so a reader can cheaply detect that another process (the other agent's stdio
server) has written and rebuild.
"""

from __future__ import annotations

import numpy as np

from local_memory.errors import LocalMemoryError
from local_memory.retrieval.fusion import DEFAULT_PREFETCH_K
from local_memory.storage import Store


class DenseIndex:
    """In-memory vector matrix mirroring the ``vectors`` table."""

    def __init__(self, store: Store, dim: int) -> None:
        self._store = store
        self.dim = dim
        self._ids: list[str] = []
        self._positions: dict[str, int] = {}
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._epoch = -1
        self.refresh_if_stale()

    # -- lifecycle ---------------------------------------------------------

    def refresh(self) -> None:
        """Unconditionally reload the matrix from the store."""
        ids, matrix = self._store.load_vectors(self.dim)
        if matrix.size and matrix.shape[1] != self.dim:
            raise LocalMemoryError(
                f"stored vectors have dim {matrix.shape[1]}, index expects {self.dim}; "
                "run `local-memory reindex`"
            )
        self._ids = ids
        self._matrix = matrix
        self._positions = {rid: i for i, rid in enumerate(ids)}
        self._epoch = self._store.epoch()

    def refresh_if_stale(self) -> bool:
        """Reload only if another process has committed since the last read."""
        if self._store.epoch() != self._epoch:
            self.refresh()
            return True
        return False

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    def contains(self, record_id: str) -> bool:
        return record_id in self._positions

    def add(self, record_id: str, vector: np.ndarray) -> None:
        """Append or replace a single vector without a full reload."""
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.dim:
            raise LocalMemoryError(f"vector has dim {vector.shape[1]}, index expects {self.dim}")
        if record_id in self._positions:
            self._matrix[self._positions[record_id]] = vector[0]
            return
        self._matrix = np.vstack([self._matrix, vector]) if len(self._ids) else vector
        self._positions[record_id] = len(self._ids)
        self._ids.append(record_id)

    def remove(self, record_id: str) -> None:
        position = self._positions.pop(record_id, None)
        if position is None:
            return
        self._ids.pop(position)
        keep = np.ones(len(self._matrix), dtype=bool)
        keep[position] = False
        self._matrix = self._matrix[keep]
        self._positions = {rid: i for i, rid in enumerate(self._ids)}

    # -- search ------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        allowed_ids: set[str] | None = None,
        limit: int = DEFAULT_PREFETCH_K,
    ) -> list[tuple[str, float]]:
        """Cosine top-k. Vectors are L2-normalised, so dot == cosine.

        ``allowed_ids`` pre-filters before the scan. Filtering first is both
        cheaper and more precise than filtering after ranking.
        """
        self.refresh_if_stale()
        if not self._ids or limit <= 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise LocalMemoryError(
                f"query has dim {query.shape[0]}, index expects {self.dim}; "
                "the model may have changed - run `local-memory reindex`"
            )

        scores = self._matrix @ query
        if allowed_ids is not None:
            mask = np.zeros(len(self._ids), dtype=bool)
            for record_id in allowed_ids:
                position = self._positions.get(record_id)
                if position is not None:
                    mask[position] = True
            if not mask.any():
                return []
            scores = np.where(mask, scores, -np.inf)

        take = min(limit, scores.shape[0])
        if take == 0:
            return []
        partition = np.argpartition(-scores, take - 1)[:take]
        partition = partition[np.argsort(-scores[partition])]
        return [(self._ids[i], float(scores[i])) for i in partition if np.isfinite(scores[i])]
