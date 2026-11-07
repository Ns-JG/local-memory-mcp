"""Embedding wrapper around fastembed.

The query prefix goes on queries only and the document prefix on documents only.
Applying the wrong one, or none, degrades retrieval without raising.

The import of fastembed is deferred so tests can inject a fake embedder and
never touch the network.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from local_memory.embedders.model_repository import EmbedderRepository
from local_memory.embedders.models import ModelSpec
from local_memory.errors import LocalMemoryError

if TYPE_CHECKING:
    from fastembed import TextEmbedding


@runtime_checkable
class Embedder(Protocol):
    """What the service needs from an embedder."""

    spec: ModelSpec

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]: ...
    def embed_query(self, text: str) -> NDArray[np.float32]: ...


class FastEmbedder:
    """Real embedder. Loads the ONNX model lazily on first use."""

    def __init__(self, spec: ModelSpec, cache_dir: Path) -> None:
        self.spec = spec
        self.cache_dir = cache_dir
        self._model: TextEmbedding | None = None

    def _ensure_loaded(self) -> TextEmbedding:
        """Load the model if it isn't already loaded."""
        from fastembed import TextEmbedding

        self._model = (
            TextEmbedding(
                model_name=self.spec.name,
                cache_dir=str(self.cache_dir),
                threads=self.spec.threads,
            )
            if self._model is None
            else self._model
        )
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._embed([self.spec.document_text(t) for t in texts])

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._embed([f"{self.spec.query_prefix}{text}"])[0]

    def _embed(self, items: list[str]) -> NDArray[np.float32]:
        if not items:
            return np.zeros((0, self.spec.dim), dtype=np.float32)
        model = self._ensure_loaded()
        vectors = self._normalise(
            np.vstack([np.asarray(v, dtype=np.float32) for v in model.embed(items)])
        )
        if vectors.shape[1] != self.spec.dim:
            raise LocalMemoryError(
                f"{self.spec.name} produced dim {vectors.shape[1]}, registry says {self.spec.dim}"
            )
        return vectors

    @classmethod
    def _normalise(cls, matrix: NDArray[np.float32]) -> NDArray[np.float32]:
        matrix = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms


def build_embedder(
    model_name: str | None, cache_dir: Path, *, threads: int | None = None
) -> FastEmbedder:
    """Factory used by the CLI and the MCP adapters."""
    spec = EmbedderRepository.resolve_model(model_name)
    if threads is not None:
        spec = replace(spec, threads=threads)
    cache_dir = cache_dir.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return FastEmbedder(spec, cache_dir)
