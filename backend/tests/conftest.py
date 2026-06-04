"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from raglearn.api.main import create_app
from raglearn.core.config import Settings, load_settings


@pytest.fixture
def config_dir() -> Path:
    """Absolute path to the repo's config directory."""
    # tests/ -> backend/ -> repo root
    return Path(__file__).resolve().parents[2] / "config"


@pytest.fixture
def settings(config_dir: Path) -> Settings:
    """Settings loaded from the dev profile."""
    return load_settings(env="dev", config_dir=config_dir)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """A TestClient backed by an app built from the dev settings."""
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
