"""Tests for the HTTP API: health and capability matrix."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["env"] == "dev"
    assert body["version"]


def test_config_matrix_reports_implementation(client: TestClient) -> None:
    response = client.get("/config")
    assert response.status_code == 200
    matrix = response.json()

    # The local cost model is implemented and active.
    cost = {a["name"]: a for a in matrix["cost_model"]["adapters"]}
    assert cost["local"]["implemented"] is True
    assert cost["local"]["active"] is True

    # The active bge-m3 embedder is implemented; its declared alternatives are
    # not yet - the matrix stays honest about what's built.
    embedder = {a["name"]: a for a in matrix["embedder"]["adapters"]}
    assert embedder["bge_m3"]["implemented"] is True
    assert embedder["bge_m3"]["active"] is True
    assert embedder["e5"]["implemented"] is False
