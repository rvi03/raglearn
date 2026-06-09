"""Flat PDF structure from the digital text layer (pypdfium2).

A rescue parser for digital PDFs whose text Docling's vision layout discards by
labeling graphic slides as pictures (e.g. investor-presentation decks).
``pypdfium2`` reads the embedded text layer directly, so the body text is
recovered — flat (one block per page, no heading/table structure), which is the
right trade when the alternative is losing the text entirely. Used only as a
fallback when Docling under-extracts a PDF that has a real text layer.
"""

from __future__ import annotations

import pypdfium2 as pdfium

from finrag.core.types import (
    BlockKind,
    ParsedStructure,
    RawDocument,
    StructureBlock,
    StructureSection,
)


def pdf_text_by_page(data: bytes) -> list[tuple[int, str]]:
    """Return ``(page_no, text)`` for each page with non-empty digital text.

    Best-effort: non-PDF bytes or any extraction error yield an empty list.
    """
    try:
        pdf = pdfium.PdfDocument(data)
    except Exception:
        return []
    try:
        pages: list[tuple[int, str]] = []
        for index in range(len(pdf)):
            text = pdf[index].get_textpage().get_text_range().strip()
            if text:
                pages.append((index + 1, text))
        return pages
    finally:
        pdf.close()


class PdfiumStructureParser:
    """Parses a PDF's digital text layer into a flat :class:`ParsedStructure`."""

    def parse(self, document: RawDocument) -> ParsedStructure:
        """Yield one untitled section whose blocks are the per-page digital text."""
        blocks = [
            StructureBlock(kind=BlockKind.TEXT, text=text, page=page)
            for page, text in pdf_text_by_page(document.data or b"")
        ]
        sections = [StructureSection(title=None, level=0, blocks=blocks)] if blocks else []
        return ParsedStructure(source_doc_id=document.doc_id, sections=sections)
