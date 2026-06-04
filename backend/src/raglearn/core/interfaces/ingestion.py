"""Ingestion-plane interfaces.

The path a document travels from a source to the indexes: connect -> extract
metadata -> dedup -> parse -> chunk -> embed.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from raglearn.core.types import (
    Chunk,
    ConnectorRequest,
    DocumentMetadata,
    EmbeddingVector,
    ParsedPage,
    RawDocument,
)


@runtime_checkable
class SourceConnector(Protocol):
    """Fetches raw documents from a source (upload, EDGAR, data.gov.in)."""

    def fetch(self, request: ConnectorRequest) -> Iterator[RawDocument]:
        """Yield raw documents for the given request."""
        ...


@runtime_checkable
class MetadataExtractor(Protocol):
    """Derives binding metadata (company, period, filing type) for a document."""

    def extract(self, document: RawDocument) -> DocumentMetadata:
        """Return metadata for the document."""
        ...


@runtime_checkable
class Intake(Protocol):
    """Content-hash dedup and versioning gate."""

    def is_new(self, document: RawDocument) -> bool:
        """Return whether the document is new and should be ingested."""
        ...


@runtime_checkable
class DocumentParser(Protocol):
    """Parses a raw document into pages, merging per-page extractor output."""

    def parse(self, document: RawDocument) -> Iterator[ParsedPage]:
        """Yield parsed pages for the document."""
        ...


@runtime_checkable
class Chunker(Protocol):
    """Splits parsed pages into structure-aware chunks carrying metadata."""

    def chunk(self, pages: Sequence[ParsedPage], metadata: DocumentMetadata) -> Iterator[Chunk]:
        """Yield chunks for the given pages."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Produces hybrid (dense + sparse) embeddings for text."""

    def embed(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """Return one embedding per input text, in order."""
        ...
