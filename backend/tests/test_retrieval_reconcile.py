"""Tests for latest-version reconciliation.

Pins the rule: among retrieved chunks, any from a superseded version of a logical
document is dropped; the newest version's chunks and any version-less chunks
survive, in their original order.
"""

from __future__ import annotations

from finrag.core.types import Chunk, ChunkType, DocumentMetadata, Market, ScoredChunk
from finrag.retrieval.reconcile import reconcile_versions


def _scored(
    chunk_id: str,
    *,
    logical_key: str | None,
    revision: int = 1,
    recency: str | None = None,
) -> ScoredChunk:
    md = DocumentMetadata(
        collection_id="c1",
        company_name="Co",
        market=Market.US,
        filing_type="10-K",
        source_doc_id=chunk_id,
        logical_key=logical_key,
        source_revision=revision,
        recency=recency,
    )
    chunk = Chunk(chunk_id=chunk_id, text="t", chunk_type=ChunkType.TEXT, metadata=md)
    return ScoredChunk(chunk=chunk, score=0.0)


def test_drops_chunks_from_a_superseded_revision() -> None:
    chunks = [
        _scored("old-1", logical_key="doc-A", revision=1),
        _scored("new-1", logical_key="doc-A", revision=2),
        _scored("old-2", logical_key="doc-A", revision=1),
    ]

    kept = [s.chunk.chunk_id for s in reconcile_versions(chunks)]

    assert kept == ["new-1"]  # only the latest revision of doc-A survives


def test_keeps_distinct_documents_and_versionless_chunks() -> None:
    chunks = [
        _scored("a", logical_key="doc-A", revision=2),
        _scored("b", logical_key="doc-B", revision=1),
        _scored("c", logical_key=None),  # nothing to reconcile against
    ]

    kept = [s.chunk.chunk_id for s in reconcile_versions(chunks)]

    assert kept == ["a", "b", "c"]  # order preserved, all retained


def test_recency_breaks_a_revision_tie() -> None:
    chunks = [
        _scored("older", logical_key="doc-A", revision=1, recency="2024-01-01"),
        _scored("newer", logical_key="doc-A", revision=1, recency="2025-01-01"),
    ]

    kept = [s.chunk.chunk_id for s in reconcile_versions(chunks)]

    assert kept == ["newer"]  # same revision → later recency wins
