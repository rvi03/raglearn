"""Route a document to the right structure parser by its detected format.

The parse lane is modality-split: SEC HTML/inline-XBRL is best read by
``sec-parser`` (visual-semantic HTML), while PDFs and images go through Docling's
vision pipeline. Plain text has no structure to recover and collapses to a single
untitled section. All lanes converge on :class:`~finrag.core.types.ParsedStructure`.

A PDF caveat: Docling's vision layout discards graphic slides (labels them
pictures), so a digital deck whose text lives in vector graphics comes back near
empty. When that happens and the PDF has a real text layer, the dispatcher falls
back to ``pypdfium2`` to recover the body text (flat, but not lost). Genuinely
scanned PDFs (no text layer) keep Docling's OCR; raster pages OCR cannot read are
the multimodal (VLM) vertical's job, not this dispatcher's.

XBRL/XML never reaches here -- it travels the facts arm (Arelle) -- and unknown
formats are quarantined upstream, so this dispatcher only sees page-lane formats.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from finrag.core.types import (
    BlockKind,
    DetectedFormat,
    ParsedStructure,
    RawDocument,
    StructureBlock,
    StructureSection,
)
from finrag.ingestion.chunking.pdfium_structure import PdfiumStructureParser
from finrag.ingestion.chunking.sec_html import SecHtmlParser

if TYPE_CHECKING:
    from finrag.ingestion.chunking.pdf_structure import DoclingStructureParser

logger = logging.getLogger(__name__)

# A PDF is "rescued" via pypdfium2 when its digital text layer is both
# substantial and far larger than what Docling extracted — i.e. Docling dropped
# most of the page as pictures. Tuned to fire on graphic decks, not normal PDFs.
_RESCUE_MIN_CHARS = 200
_RESCUE_RATIO = 2.0


def _text_len(structure: ParsedStructure) -> int:
    """Total characters of block text in a parsed structure."""
    return sum(len(block.text) for section in structure.sections for block in section.blocks)


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
        pdfium_parser: PdfiumStructureParser | None = None,
    ) -> None:
        """Bind the lane parsers.

        The Docling PDF parser is built lazily on first PDF/image (see
        :attr:`_pdf`): it pulls in Docling's vision stack (torch/torchvision), so
        deferring keeps the HTML/XBRL/text lanes — and consumer startup — free of
        that heavy import. The SEC HTML and pypdfium2 lanes are cheap, built now.
        """
        self._sec = sec_parser or SecHtmlParser()
        self._pdf_parser = pdf_parser  # lazily constructed on first PDF/image use
        self._pdfium = pdfium_parser or PdfiumStructureParser()

    @property
    def _pdf(self) -> DoclingStructureParser:
        """The Docling PDF parser, imported and constructed on first use."""
        if self._pdf_parser is None:
            from finrag.ingestion.chunking.pdf_structure import DoclingStructureParser

            self._pdf_parser = DoclingStructureParser()
        return self._pdf_parser

    def parse(self, document: RawDocument, fmt: DetectedFormat) -> ParsedStructure:
        """Parse a document into section structure using the lane for ``fmt``."""
        structure, _ = self.parse_with_decision(document, fmt)
        return structure

    def parse_with_decision(
        self, document: RawDocument, fmt: DetectedFormat
    ) -> tuple[ParsedStructure, dict[str, object]]:
        """Parse a document *and* report which lane handled it + the shape recovered.

        The decision (lane, pdfium-rescue verdict + the char counts behind it,
        sections/chars/tables) is what the monitor trace surfaces, so a developer can
        see exactly how a document was parsed — not just that it was.
        """
        if fmt is DetectedFormat.PDF:
            structure, decision = self._parse_pdf(document)
        elif fmt is DetectedFormat.IMAGE:
            structure, decision = self._pdf.parse(document), {"lane": "Docling OCR"}
        elif fmt is DetectedFormat.HTML:
            structure, decision = self._sec.parse(document), {"lane": "SEC HTML parser"}
        else:
            structure, decision = _flat_structure(document), {"lane": "plain text"}
        decision["sections"] = len(structure.sections)
        decision["chars"] = _text_len(structure)
        decision["tables"] = sum(
            1 for s in structure.sections for b in s.blocks if b.kind is BlockKind.TABLE
        )
        return structure, decision

    def _parse_pdf(self, document: RawDocument) -> tuple[ParsedStructure, dict[str, object]]:
        """Parse a PDF via Docling, rescuing the text layer with pypdfium2 if dropped.

        Docling gives richer structure (headings, tables) and wins for normal
        PDFs. But when it returns far less text than the PDF's digital text layer
        holds (a graphic-heavy deck whose slides it dropped as pictures), the flat
        pypdfium2 extraction is used instead — recovering the body text. The returned
        decision records which won and the char counts that drove it.
        """
        structure = self._pdf.parse(document)
        rescued = self._pdfium.parse(document)
        docling_chars, rescued_chars = _text_len(structure), _text_len(rescued)
        info: dict[str, object] = {"docling_chars": docling_chars, "pdfium_chars": rescued_chars}
        if rescued_chars >= _RESCUE_MIN_CHARS and rescued_chars > docling_chars * _RESCUE_RATIO:
            logger.info(
                "pdfium rescue for %s: docling=%d vs pdfium=%d chars (graphic-heavy PDF)",
                document.doc_id,
                docling_chars,
                rescued_chars,
            )
            return rescued, {"lane": "pdfium rescue", "rescued": True, **info}
        return structure, {"lane": "Docling vision", "rescued": False, **info}
