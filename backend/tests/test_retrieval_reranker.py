"""Tests for the bge-reranker adapter.

Hermetic: neither the model nor torch is loaded. The lazy loader is stubbed and
the scoring function (`_compute_scores`, which owns the tokenizer/torch forward)
is monkeypatched with canned scores, so these tests pin the adapter's pair
construction, reordering, empty handling, and registration. The real forward is
checked in an out-of-band run, not here.
"""

from __future__ import annotations

from typing import Any

import pytest

from finrag.core.interfaces.retrieval import Reranker
from finrag.core.registry import registry
from finrag.core.types import (
    Chunk,
    ChunkType,
    DocumentMetadata,
    Market,
    Query,
    ScoredChunk,
)
from finrag.retrieval import reranker as reranker_mod
from finrag.retrieval.reranker import BgeReranker


def _scored(chunk_id: str, text: str) -> ScoredChunk:
    md = DocumentMetadata(
        collection_id="c1",
        company_name="Co",
        market=Market.US,
        filing_type="10-K",
        source_doc_id="us/x.htm",
    )
    chunk = Chunk(chunk_id=chunk_id, text=text, chunk_type=ChunkType.TEXT, metadata=md)
    return ScoredChunk(chunk=chunk, score=0.0)  # incoming bi-encoder score, replaced on rerank


def _stub_scores(
    monkeypatch: pytest.MonkeyPatch, scores: list[float], capture: dict[str, Any] | None = None
) -> None:
    """Make the adapter skip model loading and return canned cross-encoder scores."""
    monkeypatch.setattr(reranker_mod, "_load_model", lambda *a, **k: ("tok", "model"))

    def fake_compute(bundle: Any, pairs: list[list[str]], **kwargs: Any) -> list[float]:
        if capture is not None:
            capture["pairs"] = pairs
            capture.update(kwargs)
        return scores

    monkeypatch.setattr(reranker_mod, "_compute_scores", fake_compute)


def test_reorders_by_descending_score(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_scores(monkeypatch, [0.1, 0.9, 0.5])
    chunks = [_scored("a", "x"), _scored("b", "y"), _scored("c", "z")]

    out = BgeReranker().rerank(Query(text="q"), chunks)

    assert [s.chunk.chunk_id for s in out] == ["b", "c", "a"]  # sorted by new score
    assert [s.score for s in out] == [0.9, 0.5, 0.1]


def test_builds_query_chunk_pairs_and_passes_params(monkeypatch: pytest.MonkeyPatch) -> None:
    capture: dict[str, Any] = {}
    _stub_scores(monkeypatch, [0.5, 0.5], capture)

    BgeReranker(batch_size=4, max_length=128).rerank(
        Query(text="how much"), [_scored("a", "alpha"), _scored("b", "beta")]
    )

    assert capture["pairs"] == [["how much", "alpha"], ["how much", "beta"]]
    assert capture["normalize"] is True
    assert capture["batch_size"] == 4
    assert capture["max_length"] == 128


def test_empty_returns_empty_without_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: Any, **kwargs: Any) -> object:
        raise AssertionError("model must not load for empty input")

    monkeypatch.setattr(reranker_mod, "_load_model", _boom)

    assert BgeReranker().rerank(Query(text="q"), []) == []


def test_registered_and_conforms_to_protocol() -> None:
    adapter = registry.create("reranker", "bge_reranker")

    assert isinstance(adapter, BgeReranker)
    assert isinstance(adapter, Reranker)
