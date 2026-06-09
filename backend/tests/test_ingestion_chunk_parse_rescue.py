"""Tests for the PDF pypdfium2 rescue in the structure dispatcher.

Hermetic: fake Docling and pypdfium2 parsers stand in, so the rescue *decision*
(use pypdfium2 when Docling under-extracts a text-layer PDF) is tested without
real parsing. The real recovery is checked by an out-of-band smoke on a deck.
"""

from __future__ import annotations

from finrag.core.types import (
    BlockKind,
    DetectedFormat,
    ParsedStructure,
    RawDocument,
    StructureBlock,
    StructureSection,
)
from finrag.ingestion.chunking.parse import StructureParser, _text_len


class _FixedParser:
    def __init__(self, structure: ParsedStructure) -> None:
        self._structure = structure

    def parse(self, document: RawDocument) -> ParsedStructure:
        return self._structure


def _struct(text: str) -> ParsedStructure:
    sections = (
        [
            StructureSection(
                title=None, level=0, blocks=[StructureBlock(kind=BlockKind.TEXT, text=text)]
            )
        ]
        if text
        else []
    )
    return ParsedStructure(source_doc_id="d", sections=sections)


def _doc() -> RawDocument:
    return RawDocument(
        doc_id="india/x/y.pdf", filename="y.pdf", content_type="application/pdf", data=b"x"
    )


def _parser(docling: str, pdfium: str) -> StructureParser:
    return StructureParser(
        pdf_parser=_FixedParser(_struct(docling)), pdfium_parser=_FixedParser(_struct(pdfium))
    )  # type: ignore[arg-type]


def test_pdf_rescued_when_docling_underextracts() -> None:
    # Graphic deck: Docling drops slides as pictures; pypdfium2 has the text layer.
    out = _parser("tiny", "x" * 500).parse(_doc(), DetectedFormat.PDF)
    assert _text_len(out) == 500


def test_pdf_keeps_docling_when_it_is_rich() -> None:
    # Normal PDF: Docling's richer structure wins; no rescue.
    out = _parser("x" * 1000, "x" * 100).parse(_doc(), DetectedFormat.PDF)
    assert _text_len(out) == 1000


def test_pdf_keeps_docling_when_no_text_layer() -> None:
    # Scanned PDF: no digital text layer to rescue with -> keep Docling (its OCR).
    out = _parser("x" * 300, "").parse(_doc(), DetectedFormat.PDF)
    assert _text_len(out) == 300


def test_image_uses_docling_and_never_rescues() -> None:
    # Raster image: no text layer; pypdfium2 is not consulted.
    out = _parser("ocr", "x" * 500).parse(_doc(), DetectedFormat.IMAGE)
    assert _text_len(out) == 3
