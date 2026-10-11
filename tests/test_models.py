"""Config, model registry, ids, hashes.

Includes a check of every registry entry against the installed fastembed. An
unverified entry is a real failure mode: it resolves, it shows up in
``local-memory models``, and it only blows up later at model-load time with a
confusing error. ``BAAI/bge-m3`` was exactly that once.
"""

from __future__ import annotations

import dataclasses
import json
import time
import uuid

import pytest

from local_memory.models import (
    DEFAULT_MODEL,
    DEFAULT_TOML,
    MODEL_REGISTRY,
    MemoryRecord,
    ModelSpec,
    Scope,
    ValidationError,
    load_config,
    new_id,
    resolve_model,
    utcnow,
    write_default_config,
)


def rec(**overrides) -> MemoryRecord:
    defaults = {
        "id": new_id(),
        "kind": "fact",
        "scope": "general",
        "title": "t",
        "gist": "g",
        "body": "b",
    }
    return MemoryRecord(**(defaults | overrides))


# -- model registry ---------------------------------------------------------


def test_every_registry_model_exists_in_fastembed() -> None:
    from fastembed import TextEmbedding

    supported = {m["model"]: m["dim"] for m in TextEmbedding.list_supported_models()}
    missing = [name for name in MODEL_REGISTRY if name not in supported]
    assert not missing, f"registry advertises models fastembed cannot load: {missing}"


def test_every_registry_dim_matches_fastembed() -> None:
    from fastembed import TextEmbedding

    supported = {m["model"]: m["dim"] for m in TextEmbedding.list_supported_models()}
    wrong = {
        name: (spec.dim, supported[name])
        for name, spec in MODEL_REGISTRY.items()
        if name in supported and spec.dim != supported[name]
    }
    assert not wrong, f"registry dims disagree with fastembed: {wrong}"


def test_default_model_is_registered() -> None:
    assert DEFAULT_MODEL in MODEL_REGISTRY
    assert resolve_model(None) is MODEL_REGISTRY[DEFAULT_MODEL]


def test_resolve_model_rejects_unknown() -> None:
    with pytest.raises(ValidationError, match="unknown model"):
        resolve_model("not/a-real-model")


def test_document_text_applies_prefix_only_when_set() -> None:
    plain = ModelSpec(name="x", dim=1)
    assert plain.document_text("body") == "body"
    prefixed = ModelSpec(name="x", dim=1, document_prefix="passage: ")
    assert prefixed.document_text("body") == "passage: body"


# -- ids --------------------------------------------------------------------


def _ms(value: str) -> int:
    """The 48-bit millisecond timestamp a UUIDv7 encodes at bits 127..80."""
    return uuid.UUID(value).int >> 80


def test_new_id_is_uuid7_with_a_time_ordered_prefix() -> None:
    ids = [new_id() for _ in range(200)]
    assert ids[0][14] == "7"  # version nibble
    assert ids[0][19] in "89ab"  # RFC 9562 variant
    assert len(set(ids)) == 200
    assert [_ms(i) for i in ids] == sorted(_ms(i) for i in ids)


def test_new_id_timestamp_advances_with_the_clock() -> None:
    """The leading hex group is ``ms >> 16`` and is stable for ~49 days.

    Reading time-ordering off the id's string prefix would be wrong; the
    timestamp only fits once you shift past the version and variant nibbles.
    """
    first = _ms(new_id())
    time.sleep(0.02)
    assert _ms(new_id()) > first


# -- hashing ----------------------------------------------------------------


def test_content_hash_is_stable_and_sensitive() -> None:
    a, b = rec(), rec()
    assert a.compute_hash() == b.compute_hash()
    b.tags.append("extra")
    assert a.compute_hash() != b.compute_hash()
    assert len(a.compute_hash()) == 64


# -- validation -------------------------------------------------------------


@pytest.mark.parametrize("field", ["title", "body", "kind", "scope"])
def test_validation_rejects_blank_required_fields(field: str) -> None:
    with pytest.raises(ValidationError):
        rec(**{field: "   "}).validate()


def test_validation_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError, match="confidence"):
        rec(confidence=1.5).validate()


def test_valid_record_passes() -> None:
    rec().validate()


# -- timestamps -------------------------------------------------------------


def test_utcnow_format() -> None:
    stamp = utcnow()
    assert stamp.endswith("Z")
    assert time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")


# -- config -----------------------------------------------------------------


def test_missing_config_falls_back_to_defaults(tmp_path) -> None:
    config = load_config(tmp_path / "absent.toml")
    assert config.dedup_threshold == 0.92
    assert config.rrf_k == 60
    assert config.scope_paths == {}


def test_round_trip_default_config(tmp_path) -> None:
    path = write_default_config(tmp_path / "config.toml")
    config = load_config(path)
    assert config.default_model == DEFAULT_MODEL
    assert config.tombstone_retention_days == 730
    assert set(config.scope_paths) == {"personal", "work", "uni"}


def test_write_default_config_does_not_clobber(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('default_model = "custom/model"', encoding="utf-8")
    write_default_config(path)
    assert path.read_text(encoding="utf-8") == 'default_model = "custom/model"'


def test_default_toml_is_valid_toml() -> None:
    import tomllib

    parsed = tomllib.loads(DEFAULT_TOML)
    assert parsed["dedup_threshold"] == 0.92
    assert "scope" in parsed


def test_missing_scope_paths_are_reported(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'[scope.work]\npaths = ["{tmp_path}/nope"]\n', encoding="utf-8")
    config = load_config(path)
    assert len(config.scope_warnings()) == 1
    assert "does not exist" in config.scope_warnings()[0]


def test_present_scope_paths_produce_no_warning(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'[scope.work]\npaths = ["{tmp_path}"]\n', encoding="utf-8")
    assert load_config(path).scope_warnings() == []


def test_scope_for_path_uses_longest_prefix(tmp_path) -> None:
    (tmp_path / "work" / "deep").mkdir(parents=True)
    (tmp_path / "general").mkdir()
    path = tmp_path / "config.toml"
    path.write_text(
        f'[scope.work]\npaths = ["{tmp_path}/work"]\n\n[scope.general]\npaths = ["{tmp_path}"]\n',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.scope_for_path(tmp_path / "work" / "deep" / "f.py") == "work"
    assert config.scope_for_path(tmp_path / "general" / "f.py") == "general"


def test_scope_for_path_falls_back_to_general(tmp_path) -> None:
    config = load_config(tmp_path / "absent.toml")
    assert config.scope_for_path(tmp_path / "whatever.py") == Scope.GENERAL


# -- redaction --------------------------------------------------------------


def test_record_serialises_to_plain_json() -> None:
    payload = json.loads(json.dumps(dataclasses.asdict(rec()), default=str))
    assert set(payload) >= {"id", "title", "body", "status"}
    assert isinstance(payload["tags"], list)
