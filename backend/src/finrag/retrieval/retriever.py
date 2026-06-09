"""Hybrid narrative retrieval: embed the query, search the vector store.

The first stage of the narrative path. The query is embedded with the *same*
bge-m3 model used at ingestion, so its dense and sparse spaces line up with the
indexed chunks; the vector store then runs its hybrid (RRF over dense + sparse)
search under whatever metadata filters the query carries. Reranking is a
separate, more precise stage layered above this one.

The embedder and vector store are injected (already built) rather than
constructed here — they are heavyweight, process-wide singletons that other
stages share.
"""

from __future__ import annotations

from finrag.core.interfaces.ingestion import Embedder
from finrag.core.interfaces.storage import VectorStore
from finrag.core.registry import registry
from finrag.core.types import Query, ScoredChunk


@registry.register("retriever", "hybrid")
class HybridRetriever:
    """A :class:`~finrag.core.interfaces.Retriever` over bge-m3 + a vector store."""

    def __init__(self, *, embedder: Embedder, vector_store: VectorStore) -> None:
        """Bind the retriever to its embedder and vector store.

        Args:
          embedder: Embeds the query into the same dense+sparse space as the index.
          vector_store: Runs the hybrid vector search.
        """
        self._embedder = embedder
        self._store = vector_store

    def retrieve(self, query: Query, *, top_k: int) -> list[ScoredChunk]:
        """Return up to ``top_k`` candidate chunks for a query.

        Args:
          query: The user question and its metadata filters.
          top_k: Maximum candidates to return (the reranker narrows from here).

        Returns:
          Scored candidate chunks, or an empty list if the query is empty.
        """
        vectors = self._embedder.embed([query.text])
        if not vectors:
            return []
        return self._store.search(
            vectors[0], top_k=top_k, filters=query.filters, access_tags=query.access_tags
        )
