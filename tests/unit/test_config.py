import dataclasses
from pathlib import Path

import pytest

from evalrag.config import ConfigError, Settings

REQUIRED = {
    "LLM_BASE_URL": "http://localhost:1/v1",
    "LLM_MODEL": "llm-x",
    "EMBED_BASE_URL": "http://localhost:2/v1",
    "EMBED_MODEL": "embed-x",
}


def test_from_env_reads_required_and_defaults():
    s = Settings.from_env(dict(REQUIRED))
    assert s.llm_base_url == "http://localhost:1/v1"
    assert s.llm_model == "llm-x"
    assert s.embed_base_url == "http://localhost:2/v1"
    assert s.embed_model == "embed-x"
    assert s.data_dir == Path("data")
    assert s.top_k == 5
    assert s.chunk_size == 200
    assert s.chunk_overlap == 40


def test_from_env_reads_optional_overrides():
    env = {**REQUIRED, "DATA_DIR": "/tmp/d", "TOP_K": "3", "CHUNK_SIZE": "50", "CHUNK_OVERLAP": "0"}
    s = Settings.from_env(env)
    assert s.data_dir == Path("/tmp/d")
    assert (s.top_k, s.chunk_size, s.chunk_overlap) == (3, 50, 0)


def test_from_env_reports_missing_key():
    for key in REQUIRED:
        env = {k: v for k, v in REQUIRED.items() if k != key}
        with pytest.raises(ConfigError, match=key):
            Settings.from_env(env)


def test_from_env_empty_required_is_error():
    env = {**REQUIRED, "LLM_MODEL": ""}
    with pytest.raises(ConfigError, match="LLM_MODEL"):
        Settings.from_env(env)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("TOP_K", "abc"),
        ("TOP_K", "0"),
        ("TOP_K", "-1"),
        ("CHUNK_SIZE", "1.5"),
        ("CHUNK_SIZE", "0"),
        ("CHUNK_SIZE", "-3"),
        ("CHUNK_OVERLAP", "x"),
        ("CHUNK_OVERLAP", "-1"),
    ],
)
def test_from_env_bad_int_names_key(key, value):
    with pytest.raises(ConfigError, match=key):
        Settings.from_env({**REQUIRED, key: value})


def test_overlap_must_be_less_than_size():
    with pytest.raises(ConfigError, match="CHUNK_OVERLAP"):
        Settings.from_env({**REQUIRED, "CHUNK_SIZE": "10", "CHUNK_OVERLAP": "10"})


def test_config_error_is_value_error():
    assert issubclass(ConfigError, ValueError)


def test_settings_is_frozen():
    s = Settings.from_env(dict(REQUIRED))
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.top_k = 9  # type: ignore[misc]
