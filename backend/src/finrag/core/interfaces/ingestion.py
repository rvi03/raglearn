"""Ingestion-plane interfaces.

The path a document travels from a source to the indexes: connect -> extract
metadata -> dedup -> parse -> chunk -> embed.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from finrag.core.types import (
    Chunk,
    ConnectorRequest,
    DetectedFormat,
    DocumentIdentity,
    DocumentMetadata,
    EmbeddingVector,
    ParsedStructure,
    RawDocument,
    XbrlExtraction,
)


@runtime_checkable
class FormatDetector(Protocol):
    """Identifies a document's format from its content, ignoring its filename."""

    def detect(self, data: bytes) -> DetectedFormat:
        """Return the coarse format of the given document bytes."""
        ...


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
class IdentityExtractor(Protocol):
    """Derives a document's content-based identity (which logical doc, which version).

    Market-specific: the dispatcher picks the adapter from the enforced
    ``filings/<country>/`` path prefix, and each adapter reads identity from the
    content its arm already produces — the US adapter from the XBRL DEI
    (``extraction``), the India adapter from the parsed cover title
    (``structure``). The filename/path below ``<country>`` is never trusted as
    identity.
    """

    def identify(
        self,
        document: RawDocument,
        *,
        structure: ParsedStructure | None = None,
        extraction: XbrlExtraction | None = None,
    ) -> DocumentIdentity:
        """Return the document's identity from its content."""
        ...


@runtime_checkable
class StructuredExtractor(Protocol):
    """Extracts a filing's structured output (metadata + XBRL facts) in one pass.

    This is the facts arm of the parse router, parallel to the pages arm
    (parse -> chunk -> embed): XBRL carries the filer's as-filed numbers and its
    DEI metadata, both pulled from a single load and bound for the structured
    store rather than parsed into chunks.
    """

    def extract(self, document: RawDocument) -> XbrlExtraction | None:
        """Return the filing's extraction, or ``None`` if its bundle is incomplete."""
        ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a parsed document's section structure into chunks carrying metadata."""

    def chunk(self, structure: ParsedStructure, metadata: DocumentMetadata) -> Iterator[Chunk]:
        """Yield chunks for the given parsed structure."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Produces hybrid (dense + sparse) embeddings for text."""

    def embed(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """Return one embedding per input text, in order."""
        ...
