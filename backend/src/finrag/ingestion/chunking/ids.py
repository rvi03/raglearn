"""Version-stable chunk identifiers.

A chunk's id keys on the document's content hash (its version) when identity has
been threaded onto the metadata: re-ingesting identical bytes then yields
identical ids (idempotent), and a changed version yields fresh ids (versions
coexist, never colliding). Before identity is wired the id falls back to the
source document id, preserving the prior behavior.
"""

from __future__ import annotations

from finrag.core.types import DocumentMetadata


def make_chunk_id(metadata: DocumentMetadata, index: int) -> str:
    """Return the chunk id for the ``index``-th chunk of a document."""
    base = metadata.content_hash[:16] if metadata.content_hash else metadata.source_doc_id
    return f"{base}-{index:04d}"
