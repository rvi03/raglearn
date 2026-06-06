"""Route a document to the right structure parser by its detected format.

The parse lane is modality-split: SEC HTML/inline-XBRL is best read by
``sec-parser`` (visual-semantic HTML), while PDFs and images go through Docling's
vision pipeline. Plain text has no structure to recover and collapses to a single
untitled section. All lanes converge on :class:`~raglearn.core.types.ParsedStructure`.

XBRL/XML never reaches here -- it travels the facts arm (Arelle) -- and unknown
formats are quarantined upstream, so this dispatcher only sees page-lane formats.
"""

from __future__ import annotations

from raglearn.core.types import (
    BlockKind,
    DetectedFormat,
    ParsedStructure,
    RawDocument,
    StructureBlock,
    StructureSection,
)
from raglearn.ingestion.chunking.pdf_structure import DoclingStructureParser
from raglearn.ingestion.chunking.sec_html import SecHtmlParser


def _decode(data: bytes) -> str:
    """Decode bytes to text, tolerating non-UTF-8 legacy filings."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _flat_structure(document: RawDocument) -> ParsedStructure:
    """Wrap a plain-text document as one untitled section (no recoverable structure)."""
    text = _decode(document.data or b"").strip()
    blocks = [StructureBlock(kind=BlockKind.TEXT, text=text)] if text else []
    section = StructureSection(title=None, level=0, blocks=blocks)
    return ParsedStructure(source_doc_id=document.doc_id, sections=[section] if blocks else [])


class StructureParser:
    """Dispatches a document to the SEC HTML, PDF, or flat-text structure parser."""

    def __init__(
        self,
        sec_parser: SecHtmlParser | None = None,
        pdf_parser: DoclingStructureParser | None = None,
    ) -> None:
        """Bind the lane parsers, constructing the defaults (both cheap to build)."""
        self._sec = sec_parser or SecHtmlParser()
        self._pdf = pdf_parser or DoclingStructureParser()

    def parse(self, document: RawDocument, fmt: DetectedFormat) -> ParsedStructure:
        """Parse a document into section structure using the lane for ``fmt``.

        Args:
          document: The fetched document.
          fmt: The format detected for the document.

        Returns:
          The document's section structure.
        """
        if fmt in (DetectedFormat.PDF, DetectedFormat.IMAGE):
            return self._pdf.parse(document)
        if fmt is DetectedFormat.HTML:
            return self._sec.parse(document)
        return _flat_structure(document)
