"""scripts/render_template.py fills the installs/ templates from the active config."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from local_memory.config import ConfigManager

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "render_template", ROOT / "scripts" / "render_template.py"
)
assert spec and spec.loader
render_template = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_template)

TEMPLATES = sorted((ROOT / "installs").glob("*.md"))


@pytest.fixture
def fill(config_manager: ConfigManager) -> dict[str, str]:
    return render_template.values(config_manager.load(), config_manager.path)


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
def test_every_placeholder_is_known_and_filled(template: Path, fill: dict[str, str]) -> None:
    out = render_template.render(template.read_text(), fill)
    assert not re.search(r"\{\{\w+\}\}", out)


def test_user_name_comes_from_the_config(fill: dict[str, str], config) -> None:
    assert fill["USER_NAME"] == config.user_name
    skill = render_template.render((ROOT / "installs" / "SKILL.md").read_text(), fill)
    assert f"Long-Term Memory about {config.user_name}" in skill


def test_scopes_and_kinds_come_from_the_config(fill: dict[str, str], config) -> None:
    assert fill["AVAILABLE_SCOPES"] == " | ".join(f'"{s}"' for s in config.scopes.available)
    assert fill["DEFAULT_SCOPE"] == config.scopes.default
    assert all(f"`{k}`" in fill["SEED_KINDS"] for k in config.seed_kinds)
    assert f"`{config.scopes.default}` → catch-all" in fill["SCOPE_MAPPINGS"]
    assert "`work` → " in fill["SCOPE_MAPPINGS"]


def test_a_multiline_value_keeps_the_indent_of_its_placeholder() -> None:
    assert (
        render_template.render("- a\n  {{X}}\n- b", {"X": "- 1\n- 2"}) == "- a\n  - 1\n  - 2\n- b"
    )


def test_an_unknown_placeholder_is_an_error() -> None:
    with pytest.raises(KeyError, match="NOPE"):
        render_template.render("{{NOPE}}", {})


def test_templates_carry_no_personal_names() -> None:
    for template in TEMPLATES:
        assert not re.search(r"NsJG|VStorm|pwr_mgr", template.read_text()), template.name
