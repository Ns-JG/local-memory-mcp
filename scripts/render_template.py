"""Fill an installs/ template with values from the active config.toml.

    render_template.py TEMPLATE [CONFIG]    writes the result to stdout

Run with the repo's venv python so `local_memory` is importable.
"""

import re
import sys
from pathlib import Path

from local_memory.config import Config, ConfigManager
from local_memory.errors import ConfigError


def _tilde(path: Path) -> str:
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


def _bar(items: list[str]) -> str:
    return " | ".join(f'"{i}"' for i in items)


def _mappings(config: Config) -> str:
    lines = []
    for name in config.scopes.available:
        paths = (
            ", ".join(f"`{_tilde(p)}/*`" for p in config.scopes.rules[name].paths)
            if name in config.scopes.rules
            else ""
        )
        default = name == config.scopes.default
        target = paths or ("catch-all, used when no path matches" if default else "no paths")
        lines.append(f"- `{name}` → {target}" + (" (default scope)" if default and paths else ""))
    return "\n".join(lines)


def values(config: Config, config_path: Path) -> dict[str, str]:
    return {
        "AVAILABLE_SCOPES": _bar(config.scopes.available),
        "USER_NAME": config.user_name,
        "DEFAULT_SCOPE": config.scopes.default,
        "SCOPE_MAPPINGS": _mappings(config),
        "SEED_KINDS": ", ".join(f"`{k}`" for k in config.seed_kinds),
        "SEED_KINDS_BAR": _bar(config.seed_kinds),
        "DEFAULT_MODEL": config.default_model,
        "DB_PATH": _tilde(config.db_path),
        "MODEL_CACHE": _tilde(config.model_cache),
        "CONFIG_PATH": _tilde(config_path),
    }


def render(template: str, fill: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        if match.group(1) not in fill:
            raise KeyError(f"unknown placeholder {match.group(0)}")
        prefix = template[template.rfind("\n", 0, match.start()) + 1 : match.start()]
        indent = "" if prefix.strip() else prefix
        return fill[match.group(1)].replace("\n", "\n" + indent)

    return re.sub(r"\{\{(\w+)\}\}", replace, template)


def main(argv: list[str]) -> int:
    manager = ConfigManager(Path(argv[2])) if len(argv) > 2 else ConfigManager()
    try:
        config = manager.load()
        sys.stdout.write(
            render(Path(argv[1]).read_text(encoding="utf-8"), values(config, manager.path))
        )
    except (ConfigError, KeyError) as exc:
        print(f"render_template: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
