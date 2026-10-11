"""Local-first hybrid-retrieval memory for coding agents.

Layers, deliberately separated so the interesting logic is testable without
MCP, without a model download, and without a network:

    models    record schema, vocabularies, model registry, config
    store     SQLite + FTS5 + vectors, optimistic writes
    embedder  fastembed wrapper
    index     dense matrix, RRF fusion
    service   search / get / store / forget / confirm  <- the whole product
    sync      export / merge bundles
    cli, mcp_stdio, mcp_http   transports
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
