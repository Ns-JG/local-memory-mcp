"""Command line interface.

`local-memory` is the operator surface: inspect, repair and script the store
without an agent in the loop. The MCP tools are the agent surface; this is for
NsJG.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Protocol

from .browse import BROWSABLE_STATUSES
from .embedder import build_embedder
from .models import (
    MODEL_REGISTRY,
    SEED_KINDS,
    Config,
    LocalMemoryError,
    Status,
    ValidationError,
    load_config,
    resolve_model,
    write_default_config,
)
from .store import Store


class _SearchHit(Protocol):
    found_by: list[str]
    id: str
    title: str
    kind: str
    scope: str
    gist: str | None


class _Service(Protocol):
    def stats(self) -> object: ...
    def search(
        self,
        query: str,
        *,
        k: int,
        scope: str | None,
        kind: str | None,
        detail: str,
    ) -> list[_SearchHit]: ...
    def get(self, record_id: str) -> object: ...


def _service(config: Config) -> _Service:
    from .server import build_service

    return build_service(config)  # type: ignore[return-value]


def _print(data: object) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


# -- commands ---------------------------------------------------------------


def cmd_config(args: argparse.Namespace, config: Config) -> int:
    path = write_default_config()
    if args.json:
        _print(
            {
                "config_path": str(path),
                "db_path": str(config.db_path),
                "model_cache": str(config.model_cache),
                "default_model": config.default_model,
                "dedup_threshold": config.dedup_threshold,
                "rrf_k": config.rrf_k,
                "tombstone_retention_days": config.tombstone_retention_days,
                "scope_paths": {
                    s: [str(p) for p in r.paths] for s, r in config.scope_paths.items()
                },
                "scope_warnings": config.scope_warnings(),
            }
        )
        return 0

    print(f"config        {path}")
    print(f"database      {config.db_path}")
    print(f"model         {config.default_model} (dim {resolve_model(config.default_model).dim})")
    print(f"model cache   {config.model_cache}")
    print(f"dedup         {config.dedup_threshold}")
    print(f"rrf k         {config.rrf_k}")
    for warning in config.scope_warnings():
        print(f"  ! {warning}", file=sys.stderr)
    return 0


def cmd_stats(args: argparse.Namespace, config: Config) -> int:
    service = _service(config)
    _print(service.stats())
    return 0


def cmd_search(args: argparse.Namespace, config: Config) -> int:
    service = _service(config)
    hits = service.search(
        args.query, k=args.k, scope=args.scope, kind=args.kind, detail=args.detail
    )
    for hit in hits:
        found = "+".join(hit.found_by)
        print(f"[{found}] {hit.id}  {hit.title}  ({hit.kind}/{hit.scope})")
        if hit.gist:
            print(f"    {hit.gist}")
    print(f"\n{len(hits)} hit(s)", file=sys.stderr)
    return 0


def cmd_get(args: argparse.Namespace, config: Config) -> int:
    service = _service(config)
    _print(_record(service.get(args.id)))
    return 0


def cmd_browse(args: argparse.Namespace, config: Config) -> int:
    """Interactive read-only record viewer.

    Goes through :mod:`local_memory.browse` rather than building a
    :class:`MemoryService`, so listing costs no model load. Search inside the
    viewer builds the service lazily on first use.
    """
    from . import browse as browse_mod

    statuses = tuple(args.status) if args.status else (Status.ACTIVE.value,)
    for value in statuses:
        if value not in browse_mod.BROWSABLE_STATUSES:
            raise ValidationError(
                f"unknown status {value!r}; choose from {', '.join(browse_mod.BROWSABLE_STATUSES)}"
            )
    if args.limit < 1:
        raise ValidationError(f"--limit must be >= 1, got {args.limit}")

    with Store(config.db_path) as store:
        return browse_mod.run_browse(
            store,
            config,
            scope=args.scope,
            kind=args.kind,
            statuses=statuses,
            limit=args.limit,
            search=args.search,
            interactive=not args.no_interactive,
        )


def cmd_reindex(args: argparse.Namespace, config: Config) -> int:
    """Recompute vectors. Required after a model change."""
    spec = resolve_model(args.model or config.default_model)
    embedder = build_embedder(spec.name, config.model_cache, threads=config.threads)
    with Store(config.db_path) as store:
        pending = (
            store.stale_records(spec.name)
            if not args.all
            else [r.id for r in store.list_records(limit=1_000_000)]
        )
        if not pending:
            print(f"nothing to do: every record is embedded with {spec.name}")
            return 0
        print(f"re-embedding {len(pending)} record(s) with {spec.name} ({spec.dim}d)")
        records = store.get_many(pending)
        vectors = embedder.embed_documents([f"{r.title}\n{r.body}" for r in records])
        store.clear_vectors()
        for record, vector in zip(records, vectors, strict=True):
            store.put_vector(record.id, vector)
            store.set_embedding_info(record.id, spec.name, spec.dim)
    print(f"done: {len(pending)} record(s) embedded with {spec.name}")
    return 0


def cmd_models(args: argparse.Namespace, config: Config) -> int:
    _print(
        {
            "default": config.default_model,
            "registry": {
                name: {"dim": s.dim, "note": s.note, "query_prefix": s.query_prefix}
                for name, s in MODEL_REGISTRY.items()
            },
        }
    )
    return 0


def cmd_kinds(args: argparse.Namespace, config: Config) -> int:
    """Show the seeded vocabulary against kinds actually present in records.

    ``kinds`` is a registry that remembers every kind ever written, including
    ones whose records have since been purged, so it is not the same thing as
    "in use". Counting from the records themselves is what an operator wants.
    """
    with Store(config.db_path) as store:
        counts: dict[str, int] = {}
        for record in store.all_records():
            counts[record.kind] = counts.get(record.kind, 0) + 1
        print("seeds:  " + ", ".join(SEED_KINDS))
        if counts:
            print("in use:")
            for kind, count in sorted(counts.items()):
                origin = "seed" if kind in SEED_KINDS else "agent-defined"
                print(f"  {kind}  ({count} record(s), {origin})")
        else:
            print("in use: (none)")
        unused = set(store.list_kinds()) - set(counts)
        if unused:
            print(f"registered but no records remain: {', '.join(sorted(unused))}")
    return 0


def cmd_purge(args: argparse.Namespace, config: Config) -> int:
    """Delete tombstones past the retention window. Irreversible."""
    import time

    with Store(config.db_path) as store:
        cutoff = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - config.tombstone_retention_days * 86400)
        )
        expired = [
            r
            for r in store.list_records(statuses=["deleted"], limit=1_000_000)
            if (r.deleted_at or "") < cutoff
        ]
        if not expired:
            print(f"no tombstones older than {config.tombstone_retention_days} days")
            return 0
        if not args.yes:
            print(f"{len(expired)} tombstone(s) eligible for purge; pass --yes to confirm")
            return 1
        for record in expired:
            store.hard_delete(record.id)
        print(f"purged {len(expired)} tombstone(s)")
    return 0


def cmd_scopes(args: argparse.Namespace, config: Config) -> int:
    if args.path:
        print(config.scope_for_path(Path(args.path)))
        return 0
    for scope, rule in config.scope_paths.items():
        print(f"{scope}:")
        for path in rule.paths:
            print(f"  {'ok ' if path.exists() else 'MISSING'} {path}")
    return 0


def cmd_doctor(args: argparse.Namespace, config: Config) -> int:
    """Check the store and the model agree. Read-only."""
    problems: list[str] = []
    spec = resolve_model(config.default_model)
    info: dict[str, object] = {
        "db_path": str(config.db_path),
        "db_exists": config.db_path.exists(),
        "model": spec.name,
        "dim": spec.dim,
        "model_cache": str(config.model_cache),
        "model_cached": any(config.model_cache.glob("**/*.onnx"))
        if config.model_cache.exists()
        else False,
    }
    with Store(config.db_path) as store:
        info["records"] = store.count()
        stale = store.stale_records(spec.name)
        info["stale_vectors"] = len(stale)
        if stale:
            problems.append(
                f"{len(stale)} record(s) are not embedded with {spec.name}; "
                "run `local-memory reindex`"
            )
        ids, matrix = store.load_vectors(spec.dim)
        info["vector_dim"] = int(matrix.shape[1]) if len(ids) else None
        if len(ids) and matrix.shape[1] != spec.dim:
            problems.append(
                f"vectors are {matrix.shape[1]}d but the model is {spec.dim}d; "
                "run `local-memory reindex --model ...`"
            )
    problems.extend(config.scope_warnings())
    info["problems"] = problems
    info["ok"] = not problems
    _print(info)
    return 0 if info["ok"] else 1


# -- helpers ----------------------------------------------------------------


def _record(record: object) -> dict[str, object]:
    return {
        "id": getattr(record, "id", None),
        "title": getattr(record, "title", None),
        "gist": getattr(record, "gist", None),
        "body": getattr(record, "body", None),
        "kind": getattr(record, "kind", None),
        "scope": getattr(record, "scope", None),
        "tags": getattr(record, "tags", None),
        "status": getattr(record, "status", None),
        "confidence": getattr(record, "confidence", None),
        "times_confirmed": getattr(record, "times_confirmed", None),
        "rev": getattr(record, "rev", None),
        "created_at": getattr(record, "created_at", None),
        "updated_at": getattr(record, "updated_at", None),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-memory",
        description="Local-first hybrid memory for coding agents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("config", help="show the resolved configuration")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("stats", help="record counts, model, kinds")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("search", help="run a hybrid search from the terminal")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--scope")
    p.add_argument("--kind")
    p.add_argument("--detail", choices=["titles", "snippets", "full"], default="snippets")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser(
        "browse",
        help="browse records interactively (read-only; falls back to a table when piped)",
    )
    p.add_argument("--scope", help="only records with this scope")
    p.add_argument("--kind", help="only records with this kind")
    p.add_argument(
        "--status",
        action="append",
        choices=list(BROWSABLE_STATUSES),
        help=(
            "only records in this status; repeatable. "
            f"default: {Status.ACTIVE.value}. "
            "note search inside browse always sees active records only."
        ),
    )
    p.add_argument("-n", "--limit", type=int, default=50, help="rows per page (default 50)")
    p.add_argument("--search", help="start with a hybrid search for this query")
    p.add_argument(
        "--no-interactive",
        action="store_true",
        help="print one table and exit, even on a terminal",
    )
    p.set_defaults(func=cmd_browse)

    p = sub.add_parser("get", help="print one record")
    p.add_argument("id")
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("reindex", help="recompute vectors (needed after a model change)")
    p.add_argument("--model", help="model to embed with; defaults to the configured one")
    p.add_argument("--all", action="store_true", help="re-embed every record, not just stale ones")
    p.set_defaults(func=cmd_reindex)

    p = sub.add_parser("models", help="list the embedding model registry")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("kinds", help="list seeded and in-use record kinds")
    p.set_defaults(func=cmd_kinds)

    p = sub.add_parser("purge", help="permanently remove expired tombstones")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_purge)

    p = sub.add_parser("scopes", help="show scope path mappings, or resolve a path")
    p.add_argument("path", nargs="?")
    p.set_defaults(func=cmd_scopes)

    p = sub.add_parser("doctor", help="check that the store and model agree")
    p.set_defaults(func=cmd_doctor)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_config()
        return int(args.func(args, config))
    except LocalMemoryError as exc:
        print(f"local-memory: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
