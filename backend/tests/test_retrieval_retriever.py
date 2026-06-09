"""Tests for the hybrid narrative retriever (hermetic: fake embedder + store).

No model loads and no Qdrant runs — a fake embedder records what it is asked to
embed and a fake store records the search call, so these tests pin the wiring:
the query text is embedded, and the resulting vector plus the query's filters and
``top_k`` are passed straight through to the vector store.
"""

from __future__ import annotations

from collections.abc import Sequence

from finrag.core.interfaces.retrieval import Retriever
from finrag.core.registry import registry
from finrag.core.types import (
    Chunk,
    ChunkType,
    DocumentMetadata,
    EmbeddingVector,
    Market,
    Query,
    ScoredChunk,
)
from finrag.retrieval.retriever import HybridRetriever


def _scored(chunk_id: str, score: float = 0.0) -> ScoredChunk:
    md = DocumentMetadata(
        collection_id="c1",
        company_name="Co",
        market=Market.US,
        filing_type="10-K",
        source_doc_id="us/x.htm",
    )
    chunk = Chunk(chunk_id=chunk_id, text="t", chunk_type=ChunkType.TEXT, metadata=md)
    return ScoredChunk(chunk=chunk, score=score)


class _FakeEmbedder:
    """Records embed calls; returns one fixed hybrid vector per text."""

    def __init__(self, *, vectors: list[EmbeddingVector] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._vectors = vectors

    def embed(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        self.calls.append(list(texts))
        if self._vectors is not None:
            return self._vectors
        return [EmbeddingVector(dense=[1.0, 0.0], sparse={1: 0.5}) for _ in texts]


class _FakeStore:
    """Records the search call; returns canned results."""

    def __init__(self, results: list[ScoredChunk]) -> None:
        self.results = results
        self.search_args: dict[str, object] | None = None

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[EmbeddingVector]) -> None:
        raise AssertionError("retriever must not upsert")

    def search(
        self,
        vector: EmbeddingVector,
        *,
        top_k: int,
        filters: dict[str, str],
        access_tags: Sequence[str] = (),
    ) -> list[ScoredChunk]:
        self.search_args = {
            "vector": vector,
            "top_k": top_k,
            "filters": filters,
            "access_tags": access_tags,
        }
        return self.results


def test_embeds_query_then_searches_with_filters_and_top_k() -> None:
    embedder = _FakeEmbedder()
    results = [_scored("a-0"), _scored("a-1")]
    store = _FakeStore(results)
    retriever = HybridRetriever(embedder=embedder, vector_store=store)

    out = retriever.retrieve(
        Query(text="net sales", filters={"collection_id": "c1"}, access_tags=["finance"]), top_k=7
    )

    assert embedder.calls == [["net sales"]]  # the query text is what gets embedded
    assert store.search_args is not None
    assert store.search_args["top_k"] == 7
    assert store.search_args["filters"] == {"collection_id": "c1"}
    assert store.search_args["access_tags"] == ["finance"]  # caller scope reaches the store
    assert store.search_args["vector"] == embedder.embed(["net sales"])[0]
    assert out == results


def test_empty_embedding_returns_empty_without_searching() -> None:
    embedder = _FakeEmbedder(vectors=[])  # nothing to search with
    store = _FakeStore([_scored("a-0")])
    retriever = HybridRetriever(embedder=embedder, vector_store=store)

    assert retriever.retrieve(Query(text="x"), top_k=5) == []
    assert store.search_args is None  # store never queried


def test_registered_and_conforms_to_protocol() -> None:
    adapter = registry.create(
        "retriever", "hybrid", embedder=_FakeEmbedder(), vector_store=_FakeStore([])
    )

    assert isinstance(adapter, HybridRetriever)
    assert isinstance(adapter, Retriever)
