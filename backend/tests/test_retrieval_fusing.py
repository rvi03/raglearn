"""Tests for the FusingRetriever composition (real RRF, fake arms).

Pins that it runs every arm, fuses with RRF (agreement wins), applies version
reconciliation to the fused set, and caps to top_k.
"""

from __future__ import annotations

from collections.abc import Sequence

from finrag.core.types import Chunk, ChunkType, DocumentMetadata, Market, Query, ScoredChunk
from finrag.retrieval.fusing_retriever import FusingRetriever
from finrag.retrieval.fusion import RrfFusion


def _scored(chunk_id: str, *, logical_key: str | None = None, revision: int = 1) -> ScoredChunk:
    md = DocumentMetadata(
        collection_id="c1",
        company_name="Co",
        market=Market.US,
        filing_type="10-K",
        source_doc_id=chunk_id,
        logical_key=logical_key,
        source_revision=revision,
    )
    chunk = Chunk(chunk_id=chunk_id, text="t", chunk_type=ChunkType.TEXT, metadata=md)
    return ScoredChunk(chunk=chunk, score=0.0)


class _FakeRetriever:
    def __init__(self, ranking: Sequence[ScoredChunk]) -> None:
        self._ranking = list(ranking)
        self.last_top_k: int | None = None

    def retrieve(self, query: Query, *, top_k: int) -> list[ScoredChunk]:
        self.last_top_k = top_k
        return self._ranking[:top_k]


def test_runs_all_arms_and_fuses() -> None:
    dense = _FakeRetriever([_scored("a"), _scored("b"), _scored("c")])
    lexical = _FakeRetriever([_scored("c"), _scored("d")])
    fusing = FusingRetriever(retrievers=[dense, lexical], fusion=RrfFusion())

    results = fusing.retrieve(Query(text="q"), top_k=10)

    assert dense.last_top_k == 10 and lexical.last_top_k == 10  # each arm ran
    assert results[0].chunk.chunk_id == "c"  # agreed-on chunk wins
    assert sorted(s.chunk.chunk_id for s in results) == ["a", "b", "c", "d"]


def test_reconciles_versions_after_fusion() -> None:
    dense = _FakeRetriever([_scored("old", logical_key="doc", revision=1)])
    lexical = _FakeRetriever([_scored("new", logical_key="doc", revision=2)])
    fusing = FusingRetriever(retrievers=[dense, lexical], fusion=RrfFusion())

    results = fusing.retrieve(Query(text="q"), top_k=10)

    assert [s.chunk.chunk_id for s in results] == ["new"]  # superseded version dropped


def test_caps_to_top_k() -> None:
    dense = _FakeRetriever([_scored("a"), _scored("b"), _scored("c")])
    fusing = FusingRetriever(retrievers=[dense], fusion=RrfFusion())

    results = fusing.retrieve(Query(text="q"), top_k=2)

    assert len(results) == 2
