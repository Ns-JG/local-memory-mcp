"""Embedding wrapper around fastembed.

Two things here are load-bearing and easy to get wrong:

* ``FASTEMBED_CACHE_PATH`` is set in ``__init__`` *before* fastembed is
  imported. fastembed otherwise caches into a temp directory, which macOS
  reaps - you would silently re-download the model every few weeks.
* The query prefix is applied to queries only. BGE-family models expect
  ``Represent this sentence for searching relevant passages:`` on the query
  side; applying it to documents, or omitting it from queries, degrades
  retrieval without raising.

The import of fastembed is deliberately deferred so tests can inject a fake
embedder and never touch the network.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .models import LocalMemoryError, ModelSpec, resolve_model


@runtime_checkable
class Embedder(Protocol):
    """What the service needs from an embedder."""

    spec: ModelSpec

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]: ...
    def embed_query(self, text: str) -> NDArray[np.float32]: ...


def configure_model_cache(cache_dir: str | os.PathLike[str]) -> str:
    """Point fastembed at a durable cache. Call before importing fastembed."""
    resolved = os.path.expanduser(str(cache_dir))
    os.makedirs(resolved, exist_ok=True)
    os.environ["FASTEMBED_CACHE_PATH"] = resolved
    return resolved


def _normalise(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class FastEmbedEmbedder:
    """Real embedder. Loads the ONNX model lazily on first use."""

    def __init__(self, spec: ModelSpec, cache_dir: str | os.PathLike[str]) -> None:
        self.spec = spec
        self.cache_dir = configure_model_cache(cache_dir)
        self._model: Any = None

    def _ensure_loaded(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding  # deferred: needs the cache path set first

            self._model = TextEmbedding(
                model_name=self.spec.name,
                cache_dir=self.cache_dir,
                threads=self.spec.threads,
            )
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        items = [self.spec.document_text(t) for t in texts]
        if not items:
            return np.zeros((0, self.spec.dim), dtype=np.float32)
        model = self._ensure_loaded()
        vectors = _normalise(np.vstack([np.asarray(v) for v in model.embed(items)]))
        if vectors.shape[1] != self.spec.dim:
            raise LocalMemoryError(
                f"{self.spec.name} produced dim {vectors.shape[1]}, registry says {self.spec.dim}"
            )
        return vectors

    def embed_query(self, text: str) -> NDArray[np.float32]:
        prefixed = f"{self.spec.query_prefix}{text}" if self.spec.query_prefix else text
        arr = self.embed_documents([prefixed])
        return arr[0:1].reshape(-1).astype(np.float32, copy=False)


def build_embedder(
    model_name: str | None, cache_dir: str | os.PathLike[str], *, threads: int | None = None
) -> FastEmbedEmbedder:
    """Factory used by the CLI and the MCP adapters."""
    spec = resolve_model(model_name)
    if threads is not None:
        spec = ModelSpec(
            name=spec.name,
            dim=spec.dim,
            query_prefix=spec.query_prefix,
            document_prefix=spec.document_prefix,
            threads=threads,
            note=spec.note,
        )
    return FastEmbedEmbedder(spec, cache_dir)
