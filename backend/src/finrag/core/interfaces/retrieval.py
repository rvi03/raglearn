"""Retrieval-plane interfaces.

Query in, ranked evidence out: transform the query, route it to path(s),
retrieve, rerank, fuse across paths, and answer exact-figure questions from the
structured store.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from finrag.core.types import FinancialFact, Query, ScoredChunk


@runtime_checkable
class QueryTransform(Protocol):
    """Rewrites / decomposes a query before retrieval."""

    def transform(self, query: Query) -> Query:
        """Return a transformed query."""
        ...


@runtime_checkable
class Router(Protocol):
    """Picks the retrieval path(s) for a query (exact, narrative, multi-hop)."""

    def route(self, query: Query) -> list[str]:
        """Return the ordered names of the paths to run."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Retrieves candidate chunks for a query (e.g. hybrid BM25 + dense)."""

    def retrieve(self, query: Query, *, top_k: int) -> list[ScoredChunk]:
        """Return up to ``top_k`` candidate chunks."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Re-scores candidate chunks against the query with a cross-encoder."""

    def rerank(self, query: Query, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Return the chunks reordered by relevance."""
        ...


@runtime_checkable
class Fusion(Protocol):
    """Merges ranked results from multiple paths (RRF + cross-path dedup)."""

    def fuse(self, results: Sequence[Sequence[ScoredChunk]]) -> list[ScoredChunk]:
        """Return a single fused, deduplicated ranking."""
        ...


@runtime_checkable
class StructuredQA(Protocol):
    """Answers exact-figure questions from the structured store."""

    def answer(self, query: Query) -> list[FinancialFact]:
        """Return the facts that answer the query."""
        ...


@runtime_checkable
class VisualRetriever(Protocol):
    """Retrieves page-image evidence for visual queries."""

    def retrieve(self, query: Query, *, top_k: int) -> list[ScoredChunk]:
        """Return up to ``top_k`` visual chunks."""
        ...
