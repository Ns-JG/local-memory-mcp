from pathlib import Path

import pytest

from local_memory.config import Bm25, ConfigManager, schema_version
from local_memory.errors import ConfigError
from local_memory.storage.schema import FTS_COLUMNS

TEMPLATE = Path(__file__).parents[1] / "config.toml"


def manager_for(tmp_path: Path, text: str) -> ConfigManager:
    path = tmp_path / "config.toml"
    path.write_text(text)
    return ConfigManager(path)


def test_shipped_template_is_valid() -> None:
    config = ConfigManager(TEMPLATE).load()
    assert config.scopes.default in config.scopes.available


def test_instructions_path_is_relative_to_the_config_file(tmp_path: Path) -> None:
    manager = manager_for(tmp_path, TEMPLATE.read_text())
    assert manager.load().instructions_path == tmp_path / "local_memory_instructions.md"
    assert ConfigManager(TEMPLATE).load().instructions_path.is_file()


def test_bm25_keys_match_the_fts_columns() -> None:
    assert tuple(Bm25.model_fields) == FTS_COLUMNS


_MISSING_KEY = ("tags = 3.0\n", "")
_EXTRA_KEY = ("tags = 3.0\n", "tags = 3.0\nextra = 1.0\n")


@pytest.mark.parametrize(("old", "new"), [_MISSING_KEY, _EXTRA_KEY])
def test_changing_the_bm25_key_set_is_an_error(tmp_path: Path, old: str, new: str) -> None:
    with pytest.raises(ConfigError):
        manager_for(tmp_path, TEMPLATE.read_text().replace(old, new)).load()


def test_bm25_weights_can_be_tuned(tmp_path: Path) -> None:
    text = TEMPLATE.read_text().replace("title = 10.0", "title = 2.5")
    assert manager_for(tmp_path, text).load().bm25.title == 2.5


def test_a_missing_key_is_reported_by_name(tmp_path: Path) -> None:
    text = TEMPLATE.read_text().replace('[server]\nhost = "127.0.0.1"\nport = 8787\n', "")
    with pytest.raises(ConfigError, match=r"server: missing"):
        manager_for(tmp_path, text).load()


def test_schema_version_is_an_int() -> None:
    assert schema_version() >= 1


def test_edits_are_seen_on_the_next_load(tmp_path: Path) -> None:
    manager = manager_for(tmp_path, TEMPLATE.read_text())
    assert "hobby" not in manager.load().scopes.available
    manager.path.write_text(
        TEMPLATE.read_text().replace('"work",', '"work", "hobby",')
        + f'\n[scopes.hobby]\npaths = ["{tmp_path}"]\n'
    )
    assert manager.load().scopes.for_path(tmp_path / "proj") == "hobby"


def test_longest_path_wins_and_default_is_the_fallback(tmp_path: Path) -> None:
    text = TEMPLATE.read_text().split("[scopes.university]")[0] + (
        f'[scopes.work]\npaths = ["{tmp_path}"]\n'
        f'[scopes.university]\npaths = ["{tmp_path / "a"}"]\n'
    )
    config = manager_for(tmp_path, text).load()
    assert config.scopes.for_path(tmp_path / "a" / "b") == "university"
    assert config.scopes.for_path(tmp_path / "c") == "work"
    assert config.scopes.for_path(tmp_path.parent) == "general"


@pytest.mark.parametrize(
    "old, new",
    [
        ("threads = 3", "threads = 0"),
        ("gist_max_chars = 200", "gist_max_chars = 0"),
        ("threads = 3", "thread = 3"),
        ('default = "general"', 'default = "nope"'),
        ("[scopes.work]", "[scopes.unlisted]"),
        ("threads = 3", "threads = [oops"),
    ],
)
def test_invalid_config_raises_config_error(tmp_path: Path, old: str, new: str) -> None:
    with pytest.raises(ConfigError):
        manager_for(tmp_path, TEMPLATE.read_text().replace(old, new)).load()


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        ConfigManager(tmp_path / "absent.toml").load()


def test_path_resolution_prefers_global_then_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ConfigManager, "GLOBAL_PATH", tmp_path / "absent.toml")
    assert ConfigManager().path == ConfigManager.REPO_PATH
    (tmp_path / "absent.toml").write_text("")
    assert ConfigManager().path == tmp_path / "absent.toml"
