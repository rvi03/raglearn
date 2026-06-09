"""Configuration loading.

Settings come from layered YAML, then environment overrides:

  config/config.yaml    base defaults (adapter menu, service URLs)
  config/config.<env>.yaml per-environment overlay (dev, cloud)
  FINRAG_* env vars    deploy-time overrides (service hosts, secrets)

The merge is explicit (no magic), and the result is validated into a typed
:class:`Settings` so a bad config fails loudly at startup rather than deep in a
request. Which adapter is active per stage lives here; turning an active name
into an instance lives in ``core.wiring``.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from finrag.core.errors import ConfigError

# FINRAG_<KEY> env var -> path into the config dict. Used by Docker Compose to
# point the API at service hostnames without editing YAML.
_ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "FINRAG_LOG_LEVEL": ("log_level",),
    "FINRAG_QDRANT_URL": ("services", "qdrant_url"),
    "FINRAG_POSTGRES_DSN": ("services", "postgres_dsn"),
    "FINRAG_REDIS_URL": ("services", "redis_url"),
    "FINRAG_OLLAMA_URL": ("services", "ollama_url"),
    "FINRAG_MINIO_ENDPOINT": ("services", "minio_endpoint"),
    "FINRAG_TIKA_URL": ("services", "tika_url"),
    "FINRAG_MINIO_ACCESS_KEY": ("services", "minio_access_key"),
    "FINRAG_MINIO_SECRET_KEY": ("services", "minio_secret_key"),
    "FINRAG_REDPANDA_BROKER": ("services", "redpanda_broker"),
    "FINRAG_INGEST_TOPIC": ("services", "ingest_topic"),
    "FINRAG_DUCKDB_PATH": ("services", "duckdb_path"),
    "FINRAG_LANGFUSE_URL": ("services", "langfuse_url"),
    # Pipeline tunable; the string value is coerced to int by Settings validation.
    "FINRAG_EMBED_BATCH_SIZE": ("ingestion", "embed_batch_size"),
}


class Env(StrEnum):
    """The environment profile a config overlay targets."""

    DEV = "dev"
    CLOUD = "cloud"


class ServiceSettings(BaseModel):
    """Connection endpoints and credentials for the backing services."""

    qdrant_url: str
    postgres_dsn: str
    redis_url: str
    ollama_url: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    redpanda_broker: str
    ingest_topic: str
    tika_url: str
    duckdb_path: str
    # Langfuse host base (e.g. http://localhost:3000); only needed when the
    # langfuse trace exporter is enabled. Empty by default.
    langfuse_url: str = ""


class StageAdapters(BaseModel):
    """The active adapter for a stage plus the menu of declared alternatives.

    ``available`` drives the capability matrix; an entry is reported as
    implemented only if an adapter is actually registered for it.
    """

    active: str
    available: list[str]


class IngestionSettings(BaseModel):
    """Tunables for the ingestion pipeline.

    ``embed_batch_size`` is the number of chunks the embedder encodes per forward
    pass. Lower it to cut peak CPU (quieter, cooler) at the cost of slower ingest;
    raise it for throughput. Defaults to a low, gentle value; override via
    ``FINRAG_EMBED_BATCH_SIZE`` or the ``ingestion`` block in config.
    """

    embed_batch_size: int = Field(default=3, ge=1)


class ObservabilitySettings(BaseModel):
    """Cross-cutting observability wiring.

    ``exporters`` selects which trace sinks the tracer fans out to, by name
    (any of ``redis``, ``langfuse``). Empty means traces are logged only.
    """

    exporters: list[str] = Field(default_factory=list)


class Settings(BaseModel):
    """Validated, fully-resolved application configuration."""

    env: Env
    log_level: str = "INFO"
    services: ServiceSettings
    adapters: dict[str, StageAdapters]
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)


def _default_config_dir() -> Path:
    """Return the repo's ``config/`` directory, resolved from this file's location.

    Used as a fallback when ``FINRAG_CONFIG_DIR`` is unset (e.g. local dev).
    In containers the env var is set explicitly, so this path is not relied on.
    """
    # .../backend/src/finrag/core/config.py -> parents[4] == repo root
    return Path(__file__).resolve().parents[4] / "config"


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dict, or return ``{}`` if it does not exist.

    Raises:
      ConfigError: If the file exists but does not parse to a mapping.
    """
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"config file is not a mapping: {path}")
    return loaded


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Nested dicts merge key-by-key; any non-dict value in ``override`` replaces
    the value in ``base``.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Overlay ``FINRAG_*`` environment variables onto the config dict."""
    for env_key, path in _ENV_OVERRIDES.items():
        value = os.getenv(env_key)
        if value is None:
            continue
        cursor = data
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[path[-1]] = value

    # The trace exporter list is a comma-separated env override (the only list
    # override), so a single process can turn exporters on without enabling them
    # for every dev process that shares the config.
    exporters = os.getenv("FINRAG_OBSERVABILITY_EXPORTERS")
    if exporters is not None:
        names = [name.strip() for name in exporters.split(",") if name.strip()]
        data.setdefault("observability", {})["exporters"] = names
    return data


def load_settings(env: str | None = None, config_dir: str | Path | None = None) -> Settings:
    """Load and validate settings for an environment.

    Resolution order: base YAML, then the ``<env>`` overlay, then ``FINRAG_*``
    environment overrides.

    Args:
      env: Environment name; defaults to ``$FINRAG_ENV`` or ``"dev"``.
      config_dir: Directory holding the config files; defaults to
        ``$FINRAG_CONFIG_DIR`` or the repo ``config/`` directory.

    Returns:
      The validated :class:`Settings`.

    Raises:
      ConfigError: If files are missing/malformed or validation fails.
    """
    env = env or os.getenv("FINRAG_ENV", "dev")
    base_dir = Path(config_dir or os.getenv("FINRAG_CONFIG_DIR") or _default_config_dir())

    base = _read_yaml(base_dir / "config.yaml")
    overlay = _read_yaml(base_dir / f"config.{env}.yaml")
    if not base and not overlay:
        raise ConfigError(f"no config files found in {base_dir}")

    data = _apply_env_overrides(_deep_merge(base, overlay))
    try:
        return Settings(**data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration for env={env!r}: {exc}") from exc
