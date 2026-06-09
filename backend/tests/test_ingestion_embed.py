"""Tests for the bge-m3 embedder adapter.

Hermetic: the heavy FlagEmbedding model is never loaded. The lazy loader is
monkeypatched with a fake encoder, so these tests exercise the adapter's output
mapping, ordering, encode arguments, and registration without a download or
torch import. Real bge-m3 output is checked in an out-of-band smoke, not here.
"""

from __future__ import annotations

from typing import Any

import pytest

from finrag.core.interfaces.ingestion import Embedder
from finrag.core.registry import registry
from finrag.ingestion import embedding
from finrag.ingestion.embedding import BgeM3Embedder


class _FakeModel:
    """Stands in for ``BGEM3FlagModel``: records encode calls, returns fixtures."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"texts": texts, **kwargs})
        n = len(texts)
        return {
            "dense_vecs": [[0.1, 0.2, 0.3] for _ in range(n)],
            "lexical_weights": [{"5": 0.7, "9": 0.3} for _ in range(n)],
        }


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> _FakeModel:
    """Replace the lazy loader so ``embed`` uses the fake instead of bge-m3."""
    model = _FakeModel()
    monkeypatch.setattr(embedding, "_load_model", lambda *args, **kwargs: model)
    return model


def test_maps_dense_and_sparse(fake_model: _FakeModel) -> None:
    [vector] = BgeM3Embedder().embed(["hello"])

    assert vector.dense == [0.1, 0.2, 0.3]
    assert vector.sparse == {5: 0.7, 9: 0.3}  # lexical-weight keys parsed str -> int


def test_preserves_order_and_count(fake_model: _FakeModel) -> None:
    vectors = BgeM3Embedder().embed(["a", "b", "c"])

    assert len(vectors) == 3
    assert all(v.dense == [0.1, 0.2, 0.3] for v in vectors)


def test_requests_dense_and_sparse_but_not_colbert(fake_model: _FakeModel) -> None:
    BgeM3Embedder(batch_size=4, max_length=256).embed(["x"])

    [call] = fake_model.calls
    assert call["return_dense"] is True
    assert call["return_sparse"] is True
    assert call["return_colbert_vecs"] is False
    assert call["batch_size"] == 4
    assert call["max_length"] == 256


def test_empty_sparse_becomes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoSparse:
        def encode(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
            return {"dense_vecs": [[1.0]], "lexical_weights": [{}]}

    monkeypatch.setattr(embedding, "_load_model", lambda *a, **k: _NoSparse())

    [vector] = BgeM3Embedder().embed(["x"])

    assert vector.sparse is None


def test_empty_input_returns_empty_without_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: Any, **kwargs: Any) -> object:
        raise AssertionError("model must not load for empty input")

    monkeypatch.setattr(embedding, "_load_model", _boom)

    assert BgeM3Embedder().embed([]) == []


def test_registered_and_conforms_to_protocol() -> None:
    adapter = registry.create("embedder", "bge_m3")

    assert isinstance(adapter, BgeM3Embedder)
    assert isinstance(adapter, Embedder)  # runtime-checkable Embedder protocol
