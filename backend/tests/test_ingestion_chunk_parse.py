"""Tests for modality-based structure-parser dispatch.

The lane parsers are stubbed, so these verify only the routing decision and the
plain-text fallback -- no sec-parser, no Docling, no real data.
"""

from __future__ import annotations

from finrag.core.types import DetectedFormat, ParsedStructure, RawDocument
from finrag.ingestion.chunking.parse import StructureParser


class _StubParser:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls = 0

    def parse(self, document: RawDocument) -> ParsedStructure:
        self.calls += 1
        return ParsedStructure(source_doc_id=self.tag, sections=[])


def _doc(data: bytes = b"x") -> RawDocument:
    return RawDocument(doc_id="d", filename="f", content_type="application/octet-stream", data=data)


def test_pdf_and_image_route_to_the_docling_parser() -> None:
    sec, pdf = _StubParser("sec"), _StubParser("pdf")
    router = StructureParser(sec_parser=sec, pdf_parser=pdf)  # type: ignore[arg-type]

    assert router.parse(_doc(), DetectedFormat.PDF).source_doc_id == "pdf"
    assert router.parse(_doc(), DetectedFormat.IMAGE).source_doc_id == "pdf"
    assert pdf.calls == 2 and sec.calls == 0


def test_html_routes_to_the_sec_parser() -> None:
    sec, pdf = _StubParser("sec"), _StubParser("pdf")
    router = StructureParser(sec_parser=sec, pdf_parser=pdf)  # type: ignore[arg-type]

    assert router.parse(_doc(), DetectedFormat.HTML).source_doc_id == "sec"
    assert sec.calls == 1 and pdf.calls == 0


def test_plain_text_becomes_one_untitled_section_without_invoking_a_parser() -> None:
    sec, pdf = _StubParser("sec"), _StubParser("pdf")
    router = StructureParser(sec_parser=sec, pdf_parser=pdf)  # type: ignore[arg-type]

    structure = router.parse(_doc(b"Just some flat legacy text."), DetectedFormat.TEXT)

    assert len(structure.sections) == 1
    assert structure.sections[0].title is None
    assert structure.sections[0].blocks[0].text == "Just some flat legacy text."
    assert sec.calls == 0 and pdf.calls == 0


def test_empty_text_yields_no_sections() -> None:
    router = StructureParser(sec_parser=_StubParser("sec"), pdf_parser=_StubParser("pdf"))  # type: ignore[arg-type]

    assert router.parse(_doc(b"   "), DetectedFormat.TEXT).sections == []
