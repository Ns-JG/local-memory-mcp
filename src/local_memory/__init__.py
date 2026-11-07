"""Local-first hybrid-retrieval memory for coding agents.

Components, each usable and testable without the ones above it:

    config, errors, models, utils    leaf modules
    embedders/                       fastembed wrapper and model registry
    storage/                         SQLite: schema + migrations, FTS5 lexical search, records
    retrieval/                       dense index and rank fusion
    service                          search / get / store / forget / confirm  <- the product
    app                              composition root: config -> store -> embedder -> service
    mcp/, cli, browse/               adapters over the service (browse reads storage directly)
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
