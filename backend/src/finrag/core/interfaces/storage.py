"""Storage-plane interfaces.

Where indexed content lives: a vector store for narrative chunks and a graph
index for entity relationships.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from finrag.core.types import Chunk, EmbeddingVector, ScoredChunk, XbrlExtraction


@runtime_checkable
class StructuredStore(Protocol):
    """Persists a filing's structured output (collection, filing, facts) to DuckDB."""

    def write(self, extraction: XbrlExtraction) -> int:
        """Write the collection, filing, and facts atomically; return facts written.

        Idempotent: re-ingesting a filing leaves existing rows untouched.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Stores chunk embeddings and searches them by vector + metadata filter."""

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[EmbeddingVector]) -> None:
        """Insert or update chunks with their embeddings."""
        ...

    def search(
        self,
        vector: EmbeddingVector,
        *,
        top_k: int,
        filters: dict[str, str],
        access_tags: Sequence[str] = (),
    ) -> list[ScoredChunk]:
        """Return the top-k chunks for a query vector under a metadata filter.

        Results are scoped to chunks the caller may see: public (untagged) chunks
        plus any carrying one of ``access_tags``.
        """
        ...

    def scroll(self) -> list[Chunk]:
        """Return every stored chunk (no vectors).

        Used to seed an in-memory lexical (BM25) index from the same corpus the
        vector store holds, so the two retrievers fuse over identical content.
        """
        ...


@runtime_checkable
class DedupStore(Protocol):
    """Tracks which document versions (by content hash) have been ingested.

    Lets the pipeline skip re-ingesting identical bytes (a re-upload or a renamed
    copy), which is what makes ingestion idempotent on the expensive pages arm
    (parse + embed). A changed version has a different hash, so it is not skipped
    and coexists.
    """

    def is_ingested(self, content_hash: str) -> bool:
        """Return whether a document with this content hash was already ingested."""
        ...

    def mark_ingested(self, content_hash: str, doc_id: str) -> None:
        """Record that a document with this content hash has been ingested."""
        ...


@runtime_checkable
class QuarantineStore(Protocol):
    """Records documents set aside for review instead of ingested.

    A document whose format the detector does not recognize is never silently
    dropped; it is quarantined with the reason so it stays visible and can be
    re-examined or hand-tagged later.
    """

    def quarantine(self, doc_id: str, detected_format: str, reason: str) -> None:
        """Record a quarantined document."""
        ...


@runtime_checkable
class GraphIndex(Protocol):
    """Entity/relationship index for multi-hop retrieval."""

    def neighbors(self, entity: str, *, depth: int) -> list[str]:
        """Return entities related to ``entity`` within ``depth`` hops."""
        ...
