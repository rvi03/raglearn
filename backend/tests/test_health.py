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

    # The local cost model is the one adapter actually implemented so far.
    cost = {a["name"]: a for a in matrix["cost_model"]["adapters"]}
    assert cost["local"]["implemented"] is True
    assert cost["local"]["active"] is True

    # Everything else is declared but not yet implemented - the matrix is honest.
    embedder = {a["name"]: a for a in matrix["embedder"]["adapters"]}
    assert embedder["bge_m3"]["implemented"] is False
