"""Environment-driven settings."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


def _required(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "")
    if not value.strip():
        raise ConfigError(f"missing required setting: {key}")
    return value


def _endpoint(env: Mapping[str, str], key: str, required: bool) -> str:
    return _required(env, key) if required else env.get(key, "")


def _int(env: Mapping[str, str], key: str, default: int, minimum: int) -> int:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_model: str
    embed_base_url: str
    embed_model: str
    data_dir: Path = Path("data")
    top_k: int = 5
    chunk_size: int = 200
    chunk_overlap: int = 40

    @classmethod
    def from_env(cls, env: Mapping[str, str], require_endpoints: bool = True) -> "Settings":
        settings = cls(
            llm_base_url=_endpoint(env, "LLM_BASE_URL", require_endpoints),
            llm_model=_endpoint(env, "LLM_MODEL", require_endpoints),
            embed_base_url=_endpoint(env, "EMBED_BASE_URL", require_endpoints),
            embed_model=_endpoint(env, "EMBED_MODEL", require_endpoints),
            data_dir=Path(env.get("DATA_DIR") or "data"),
            top_k=_int(env, "TOP_K", 5, 1),
            chunk_size=_int(env, "CHUNK_SIZE", 200, 1),
            chunk_overlap=_int(env, "CHUNK_OVERLAP", 40, 0),
        )
        if settings.chunk_overlap >= settings.chunk_size:
            raise ConfigError(
                f"CHUNK_OVERLAP ({settings.chunk_overlap}) must be less than "
                f"CHUNK_SIZE ({settings.chunk_size})"
            )
        return settings
