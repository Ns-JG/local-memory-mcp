"""Composition root: the one place that wires config, store, embedder and service."""

from __future__ import annotations

from local_memory.config import ConfigManager
from local_memory.embedders import build_embedder
from local_memory.service import MemoryService
from local_memory.storage import Store


def build_service(manager: ConfigManager | None = None) -> MemoryService:
    manager = manager or ConfigManager()
    config = manager.load()
    embedder = build_embedder(config.default_model, config.model_cache, threads=config.threads)
    return MemoryService(store=Store(config.db_path), embedder=embedder, config=manager.load)
