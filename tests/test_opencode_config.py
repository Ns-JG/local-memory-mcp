"""scripts/opencode_config.py edits the opencode config as text."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "opencode_config.py"
spec = importlib.util.spec_from_file_location("opencode_config", SCRIPT)
assert spec and spec.loader
opencode_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opencode_config)

ENTRY = "/x/local_memory_mcp_instructions.md"


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return tmp_path / "opencode.jsonc"


def test_add_appends_to_an_existing_array_and_keeps_comments(config: Path) -> None:
    config.write_text('{\n  // keep me\n  "instructions": [\n    "/other.md"\n  ]\n}\n')
    assert opencode_config.add(config, ENTRY) == 0
    text = config.read_text()
    assert "// keep me" in text
    assert f'"/other.md",\n    "{ENTRY}"' in text


def test_add_is_idempotent(config: Path) -> None:
    config.write_text('{\n  "instructions": []\n}\n')
    opencode_config.add(config, ENTRY)
    once = config.read_text()
    opencode_config.add(config, ENTRY)
    assert config.read_text() == once


def test_add_creates_the_array_when_missing(config: Path) -> None:
    config.write_text('{\n  "mcp": {}\n}\n')
    opencode_config.add(config, ENTRY)
    assert f'"instructions": [\n    "{ENTRY}"\n  ],' in config.read_text()


def test_add_ignores_brackets_inside_strings(config: Path) -> None:
    config.write_text('{\n  "instructions": [\n    "/a[1].md"\n  ]\n}\n')
    opencode_config.add(config, ENTRY)
    assert f'"/a[1].md",\n    "{ENTRY}"' in config.read_text()


def test_add_handles_an_escaped_quote_before_a_bracket(config: Path) -> None:
    existing = r'"/a\"]b.md"'
    config.write_text('{\n  "instructions": [\n    ' + existing + "\n  ]\n}\n")
    opencode_config.add(config, ENTRY)
    assert existing + ',\n    "' + ENTRY + '"' in config.read_text()


@pytest.mark.parametrize(
    "before",
    [
        f'{{"instructions": ["/other.md", "{ENTRY}"]}}',
        f'{{"instructions": ["{ENTRY}", "/other.md"]}}',
        f'{{"instructions": ["{ENTRY}"]}}',
    ],
)
def test_remove_takes_only_its_entry(config: Path, before: str) -> None:
    config.write_text(before)
    opencode_config.remove(config, ENTRY)
    assert ENTRY not in config.read_text()
    assert ("/other.md" in config.read_text()) == ("/other.md" in before)


def test_unknown_command_is_a_usage_error() -> None:
    assert opencode_config.main(["opencode_config.py", "nope", "a", "b"]) == 2
