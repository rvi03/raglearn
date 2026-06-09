"""Latest-version reconciliation over retrieved chunks.

Ingestion GC already keeps only the latest version of a filing per ``logical_key``,
so in steady state one version is indexed. This is the query-time backstop for the
window where that is not true — a restatement mid-ingest, an index that retained
versions deliberately, or two paths disagreeing — so a superseded version can
never contribute evidence to an answer.

The rule is simple and provenance-driven: group retrieved chunks by their logical
document (``logical_key``), find the newest version present (``source_revision``,
then ``recency`` as a tiebreak), and drop every chunk that came from an older one.
Chunks without a ``logical_key`` are passed through untouched — there is nothing to
reconcile them against.
"""

from __future__ import annotations

from collections.abc import Sequence

from finrag.core.types import ScoredChunk


def _version_key(scored: ScoredChunk) -> tuple[int, str]:
    """Order versions of one logical document: higher revision, then later recency."""
    metadata = scored.chunk.metadata
    return (metadata.source_revision, metadata.recency or "")


def reconcile_versions(chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """Drop chunks belonging to a superseded version of their logical document.

    Args:
      chunks: Retrieved (and possibly fused) chunks, in ranked order.

    Returns:
      The same chunks in the same order, minus any whose ``logical_key`` has a
      newer version also present. Chunks without a ``logical_key`` are kept.
    """
    latest: dict[str, tuple[int, str]] = {}
    for scored in chunks:
        key = scored.chunk.metadata.logical_key
        if key is None:
            continue
        version = _version_key(scored)
        if key not in latest or version > latest[key]:
            latest[key] = version

    kept: list[ScoredChunk] = []
    for scored in chunks:
        key = scored.chunk.metadata.logical_key
        if key is None or _version_key(scored) == latest[key]:
            kept.append(scored)
    return kept
