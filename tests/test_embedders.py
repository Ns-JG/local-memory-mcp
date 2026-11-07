"""Model registry and FastEmbedder. A fake ``fastembed`` keeps these offline."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from local_memory.embedders import EmbedderRepository, FastEmbedder, ModelSpec, build_embedder
from local_memory.errors import LocalMemoryError, ValidationError

SPEC = ModelSpec(name="fake/model", dim=4, query_prefix="q: ", document_prefix="p: ")


class FakeTextEmbedding:
    loads = 0
    seen: list[str] = []
    dim = 4

    def __init__(self, model_name: str, cache_dir: str, threads: int) -> None:
        type(self).loads += 1

    def embed(self, items: list[str]):
        type(self).seen.extend(items)
        return (np.ones(type(self).dim, dtype=np.float32) for _ in items)


@pytest.fixture
def fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> type[FakeTextEmbedding]:
    fake = type("Fake", (FakeTextEmbedding,), {"loads": 0, "seen": [], "dim": 4})
    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=fake))
    return fake


@pytest.fixture
def embedder(fake_fastembed, tmp_path) -> FastEmbedder:
    return FastEmbedder(SPEC, tmp_path)


# -- embedder ---------------------------------------------------------------


def test_model_is_loaded_once_across_calls(fake_fastembed, embedder: FastEmbedder) -> None:
    embedder.embed_documents(["a"])
    embedder.embed_documents(["b"])
    embedder.embed_query("c")
    assert fake_fastembed.loads == 1


def test_empty_input_does_not_load_the_model(fake_fastembed, embedder: FastEmbedder) -> None:
    assert embedder.embed_documents([]).shape == (0, 4)
    assert fake_fastembed.loads == 0


def test_documents_get_the_document_prefix_only(fake_fastembed, embedder: FastEmbedder) -> None:
    embedder.embed_documents(["hello"])
    assert fake_fastembed.seen == ["p: hello"]


def test_queries_get_the_query_prefix_only(fake_fastembed, embedder: FastEmbedder) -> None:
    embedder.embed_query("hello")
    assert fake_fastembed.seen == ["q: hello"]


def test_vectors_are_unit_length(embedder: FastEmbedder) -> None:
    assert np.linalg.norm(embedder.embed_query("x")) == pytest.approx(1.0)


def test_dimension_mismatch_with_registry_is_an_error(
    fake_fastembed, embedder: FastEmbedder
) -> None:
    fake_fastembed.dim = 7
    with pytest.raises(LocalMemoryError, match="registry says 4"):
        embedder.embed_documents(["x"])


def test_build_embedder_applies_threads_and_creates_the_cache_dir(tmp_path) -> None:
    built = build_embedder(None, tmp_path / "models", threads=7)
    assert built.spec.threads == 7
    assert (tmp_path / "models").is_dir()
    assert EmbedderRepository.resolve_model(None).threads == 3


# -- registry ---------------------------------------------------------------


def test_default_model_is_registered() -> None:
    registry = EmbedderRepository.get_model_registry()
    assert EmbedderRepository.resolve_model(None) is registry[EmbedderRepository.DEFAULT_MODEL]


def test_resolve_model_rejects_unknown() -> None:
    with pytest.raises(ValidationError, match="unknown model"):
        EmbedderRepository.resolve_model("not/a-real-model")


def test_registry_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        EmbedderRepository()


def test_document_text_applies_prefix_only_when_set() -> None:
    assert ModelSpec(name="x", dim=1).document_text("body") == "body"
    assert ModelSpec(name="x", dim=1, document_prefix="p: ").document_text("body") == "p: body"


def test_every_registry_model_exists_in_fastembed_with_the_same_dim() -> None:
    from fastembed import TextEmbedding

    supported = {m["model"]: m["dim"] for m in TextEmbedding.list_supported_models()}
    registry = EmbedderRepository.get_model_registry()
    assert not [n for n in registry if n not in supported]
    assert not {n: (s.dim, supported[n]) for n, s in registry.items() if s.dim != supported[n]}
