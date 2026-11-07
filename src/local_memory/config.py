"""Configuration: the typed shape of config.toml and a reader for it."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from local_memory.errors import ConfigError


class ScopeRule(BaseModel):
    """A [scopes.<name>] table: the directories that belong to one scope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    paths: list[Path] = []

    @field_validator("paths")
    @classmethod
    def _expand(cls, paths: list[Path]) -> list[Path]:
        return [p.expanduser().resolve() for p in paths]


class Scopes(BaseModel):
    """The [scopes] table: fixed keys plus one [scopes.<name>] table per scope."""

    model_config = ConfigDict(frozen=True, extra="allow")

    available: list[str]
    default: str
    __pydantic_extra__: dict[str, ScopeRule] = Field(init=False)  # type: ignore

    @model_validator(mode="after")
    def _declared(self) -> Scopes:
        if self.default not in self.available:
            raise ValueError(f"default {self.default!r} is not in available")
        if undeclared := set(self.rules) - set(self.available):
            raise ValueError(f"[scopes.*] tables not listed in available: {sorted(undeclared)}")
        return self

    @property
    def rules(self) -> dict[str, ScopeRule]:
        return self.__pydantic_extra__

    def for_path(self, path: Path) -> str:
        """Longest matching scope path wins; ``default`` if none match."""
        resolved = path.expanduser().resolve()
        matches = [
            (len(root.parts), name)
            for name, rule in self.rules.items()
            for root in rule.paths
            if resolved == root or root in resolved.parents
        ]
        return max(matches)[1] if matches else self.default


class Server(BaseModel):
    """The [server] table: where the HTTP transport listens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str
    port: int = Field(ge=1, le=65535)


class Bm25(BaseModel):
    """The [bm25] table: one weight per FTS5 column. Adding or dropping a key is an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: float = Field(ge=0, allow_inf_nan=False)
    gist: float = Field(ge=0, allow_inf_nan=False)
    body: float = Field(ge=0, allow_inf_nan=False)
    tags: float = Field(ge=0, allow_inf_nan=False)


class Config(BaseModel):
    """System-wide configuration for local-memory, read from config.toml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    user_name: str = Field(min_length=1)
    db_path: Path
    model_cache: Path
    default_model: str
    threads: int = Field(ge=1)
    dedup_threshold: float = Field(gt=0, le=1)
    rrf_k: int = Field(gt=0)
    tombstone_retention_days: int = Field(gt=0)
    gist_max_chars: int = Field(gt=0)
    instructions_path: Path
    seed_kinds: list[str]
    server: Server
    bm25: Bm25
    scopes: Scopes

    def scope_warnings(self) -> list[str]:
        """Scope paths that do not exist on disk."""
        return [
            f"scope {name!r} maps to {path} which does not exist"
            for name, rule in self.scopes.rules.items()
            for path in rule.paths
            if not path.exists()
        ]

    @field_validator("db_path", "model_cache")
    @classmethod
    def _expand(cls, path: Path) -> Path:
        return path.expanduser()

    @field_validator("instructions_path")
    @classmethod
    def _beside_config(cls, path: Path, info: ValidationInfo) -> Path:
        config_dir = info.context.get("dir") if info.context is not None else None
        return (config_dir or Path.cwd()) / path.expanduser()


class ConfigManager:
    """Reads and validates config.toml."""

    GLOBAL_PATH = Path("~/.config/local-memory/config.toml").expanduser()
    REPO_PATH = Path(__file__).resolve().parents[2] / "config.toml"

    def __init__(self, path: Path | None = None) -> None:
        """Explicit path, else the global config, else the repo template."""
        self.path = path or next(
            (p for p in (self.GLOBAL_PATH, self.REPO_PATH) if p.is_file()), self.GLOBAL_PATH
        )

    def load(self) -> Config:
        try:
            return Config.model_validate(
                tomllib.loads(self.path.read_text(encoding="utf-8")),
                context={"dir": self.path.parent},
            )
        except ValidationError as exc:
            problems = "".join(
                f"\n  {'.'.join(map(str, e['loc']))}: "
                f"{'missing' if e['type'] == 'missing' else e['msg']}"
                for e in exc.errors()
            )
            raise ConfigError(f"{self.path}:{problems}") from exc
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"{self.path}: {exc}") from exc


def schema_version() -> int:
    return int((ConfigManager.REPO_PATH.parent / "SCHEMA_VERSION").read_text())
