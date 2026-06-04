"""Storage-plane interfaces.

Where indexed content lives: a vector store for narrative chunks and a graph
index for entity relationships.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from raglearn.core.types import Chunk, EmbeddingVector, ScoredChunk


@runtime_checkable
class VectorStore(Protocol):
    """Stores chunk embeddings and searches them by vector + metadata filter."""

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[EmbeddingVector]) -> None:
        """Insert or update chunks with their embeddings."""
        ...

    def search(
        self, vector: EmbeddingVector, *, top_k: int, filters: dict[str, str]
    ) -> list[ScoredChunk]:
        """Return the top-k chunks for a query vector under a metadata filter."""
        ...


@runtime_checkable
class GraphIndex(Protocol):
    """Entity/relationship index for multi-hop retrieval."""

    def neighbors(self, entity: str, *, depth: int) -> list[str]:
        """Return entities related to ``entity`` within ``depth`` hops."""
        ...
