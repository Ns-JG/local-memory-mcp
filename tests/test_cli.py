from __future__ import annotations

import sys
from pathlib import Path

from local_memory import cli
from local_memory.config import ConfigManager


def test_export_config_copies_the_active_config(tmp_path: Path, monkeypatch, capsys) -> None:
    source = tmp_path / "config.toml"
    source.write_text(ConfigManager.REPO_PATH.read_text())
    monkeypatch.setattr(ConfigManager, "GLOBAL_PATH", source)
    out = tmp_path / "exported.toml"
    monkeypatch.setattr(sys, "argv", ["local-memory", "export-config", str(out)])
    assert cli.main() == 0
    assert out.read_text() == source.read_text()


def _scopes_output(tmp_path: Path, monkeypatch, capsys, *, exists: bool) -> str:
    config = tmp_path / "config.toml"
    config.write_text(ConfigManager.REPO_PATH.read_text().replace("~/Desktop/work", str(tmp_path)))
    if not exists:
        config.write_text(config.read_text().replace(str(tmp_path), str(tmp_path / "gone")))
    monkeypatch.setattr(ConfigManager, "GLOBAL_PATH", config)
    monkeypatch.setattr(sys, "argv", ["local-memory", "scopes"])
    assert cli.main() == 0
    return capsys.readouterr().out


def test_output_is_plain_when_not_a_terminal(tmp_path: Path, monkeypatch, capsys) -> None:
    out = _scopes_output(tmp_path, monkeypatch, capsys, exists=False)
    assert "MISSING" in out
    assert "\x1b" not in out


def test_missing_is_red_and_ok_is_green_when_colour_is_forced(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setattr(cli, "_out", cli.Console(highlight=False, markup=False))
    assert "\x1b[31mMISSING" in _scopes_output(tmp_path, monkeypatch, capsys, exists=False)
    assert "\x1b[32mok " in _scopes_output(tmp_path, monkeypatch, capsys, exists=True)


def test_export_config_to_a_missing_directory_is_a_clean_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "config.toml"
    source.write_text(ConfigManager.REPO_PATH.read_text())
    monkeypatch.setattr(ConfigManager, "GLOBAL_PATH", source)
    out = tmp_path / "no" / "such" / "dir" / "x.toml"
    monkeypatch.setattr(sys, "argv", ["local-memory", "export-config", str(out)])
    assert cli.main() == 1
    assert "cannot export" in capsys.readouterr().err
    assert not out.exists()
