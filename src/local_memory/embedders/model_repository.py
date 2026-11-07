"""Registry of supported embedding models.

Every entry is verified against ``fastembed.TextEmbedding.list_supported_models()``
by ``tests/test_embedders.py`` (name and dim). An unverified entry resolves and
shows up in listings, then only fails later at model-load time.
"""

from local_memory.embedders.models import ModelSpec
from local_memory.errors import ValidationError


class EmbedderRepository:
    """A static registry of embedding models and their properties.

    Operates purely at class scope — do not instantiate.
    """

    # --- Prefixes ---
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    E5_QUERY_PREFIX = "query: "
    E5_DOCUMENT_PREFIX = "passage: "

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    _model_registry: dict[str, ModelSpec] = {
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

    def __init__(self) -> None:
        raise TypeError("EmbedderRepository is a class-level registry, do not instantiate.")

    @classmethod
    def get_model_registry(cls) -> dict[str, ModelSpec]:
        """Getter for the model registry dictionary."""
        return cls._model_registry

    @classmethod
    def resolve_model(cls, name: str | None = None) -> ModelSpec:
        """Look up a model by name, or the default when ``name`` is None."""
        key: str = name or cls.DEFAULT_MODEL
        if key not in cls._model_registry:
            available_models = sorted(cls._model_registry.keys())
            raise ValidationError(f"unknown model {key!r}; known: {', '.join(available_models)}")
        return cls._model_registry[key]
