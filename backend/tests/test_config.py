"""Tests for layered configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from raglearn.core.config import Env, load_settings
from raglearn.core.errors import ConfigError


def test_dev_profile_loads(config_dir: Path) -> None:
    settings = load_settings(env="dev", config_dir=config_dir)
    assert settings.env is Env.DEV
    assert settings.log_level == "DEBUG"  # overlay overrides base INFO
    assert settings.adapters["cost_model"].active == "local"
    assert settings.adapters["embedder"].active == "bge_m3"


def test_base_services_inherited(config_dir: Path) -> None:
    settings = load_settings(env="dev", config_dir=config_dir)
    assert settings.services.qdrant_url.startswith("http")
    assert "postgresql://" in settings.services.postgres_dsn


def test_env_override_applied(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGLEARN_QDRANT_URL", "http://raglearn-qdrant:6333")
    settings = load_settings(env="dev", config_dir=config_dir)
    assert settings.services.qdrant_url == "http://raglearn-qdrant:6333"


def test_missing_config_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings(env="dev", config_dir=tmp_path)
