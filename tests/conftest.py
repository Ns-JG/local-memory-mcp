"""Shared fixtures.

The fake embedder is a hashed bag of words rather than a stub returning
constants: dedup and ranking are cosine-similarity behaviours, so a fake that
returns fixed vectors would let those tests pass while the real code was
broken. Hashing is done with hashlib, not the builtin ``hash``, which is salted
per process and would make vectors non-reproducible across runs.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from local_memory.config import Config, ConfigManager
from local_memory.embedders import Embedder, ModelSpec
from local_memory.service import MemoryService
from local_memory.storage import Store

TEMPLATE = Path(__file__).parents[1] / "config.toml"

_TOKEN = re.compile(r"\w+")
FAKE_DIM = 256
FAKE_PREFIX = "Q: "


def _token_vector(text: str, dim: int) -> np.ndarray:
    """Hashed unigrams + bigrams.

    Bigrams matter: unigrams alone put two unrelated sentences in the same few
    buckets at this width, which pushed their cosine over the dedup threshold
    and made unrelated records collapse into one. Bigrams keep genuine
    paraphrases separable while identical text still scores exactly 1.0.
    """
    tokens = _TOKEN.findall(text.lower())
    vector = np.zeros(dim, dtype=np.float32)
    features = list(tokens) + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
    for feature in features:
        slot = int(hashlib.blake2b(feature.encode(), digest_size=4).hexdigest(), 16) % dim
        vector[slot] += 1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


class FakeEmbedder:
    """Deterministic, seedless, offline. Satisfies the Embedder protocol."""

    def __init__(self, dim: int = FAKE_DIM, prefix: str = FAKE_PREFIX) -> None:
        self.spec = ModelSpec(
            name="fake/hashed-bow", dim=dim, query_prefix=prefix, document_prefix="D: "
        )

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.spec.dim), dtype=np.float32)
        return np.vstack([_token_vector(self.spec.document_text(t), self.spec.dim) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return _token_vector(f"{self.spec.query_prefix}{text}", self.spec.dim)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def config_manager(tmp_path) -> ConfigManager:
    """The shipped template plus a ``personal`` scope, in a throwaway file."""
    text = TEMPLATE.read_text().replace('"work",', '"work",\n    "personal",', 1)
    path = tmp_path / "config.toml"
    path.write_text(text)
    shutil.copy(TEMPLATE.with_name("local_memory_instructions.md"), tmp_path)
    return ConfigManager(path)


@pytest.fixture
def config(config_manager: ConfigManager) -> Config:
    return config_manager.load()


@pytest.fixture
def store(tmp_path) -> Iterator[Store]:
    with Store(tmp_path / "memory.db") as opened:
        yield opened


@pytest.fixture
def service(store: Store, embedder: FakeEmbedder, config_manager: ConfigManager) -> MemoryService:
    return MemoryService(store, embedder, config_manager.load)


def make_embedder() -> Embedder:
    return FakeEmbedder()
