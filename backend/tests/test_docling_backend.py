"""Tests for the Docling parser backend.

Three layers, none of which download OCR/table models:
- the locked vision pipeline config is asserted field by field;
- the document -> pages mapping is checked against a fake document so the
  per-page (PDF) branch is covered without a real PDF;
- a real HTML conversion runs end to end (the structural pipeline needs no
  models), exercising DoclingParser against live Docling.
"""

from __future__ import annotations

from docling.datamodel.pipeline_options import AcceleratorDevice, TableFormerMode

from raglearn.core.types import RawDocument
from raglearn.ingestion.docling_backend import (
    DoclingParser,
    _vision_pipeline_options,
    build_docling_parser,
    document_to_pages,
)


def test_vision_options_match_the_locked_cpu_first_config() -> None:
    opts = _vision_pipeline_options()

    assert opts.do_ocr is True
    assert opts.generate_page_images is False
    assert opts.ocr_options.lang == ["english"]
    assert opts.ocr_options.backend == "onnxruntime"
    assert opts.ocr_options.force_full_page_ocr is False
    assert opts.ocr_options.bitmap_area_threshold == 0.05
    assert opts.do_table_structure is True
    assert opts.table_structure_options.mode is TableFormerMode.ACCURATE
    assert opts.table_structure_options.do_cell_matching is True
    assert opts.accelerator_options.device is AcceleratorDevice.CPU
    assert opts.accelerator_options.num_threads >= 1


# --- document -> pages mapping (fakes, no real conversion) --------------------


class _FakeProv:
    def __init__(self, page_no: int) -> None:
        self.page_no = page_no


class _FakeTable:
    def __init__(self, page_no: int, markdown: str) -> None:
        self.prov = [_FakeProv(page_no)]
        self._markdown = markdown

    def export_to_markdown(self, doc: object) -> str:
        return self._markdown


class _FakeDoc:
    def __init__(self, pages: dict[int, object], tables: list[_FakeTable]) -> None:
        self.pages = pages
        self.tables = tables

    def export_to_markdown(self, page_no: int | None = None) -> str:
        return "whole" if page_no is None else f"page-{page_no}"


def test_mapping_emits_one_page_each_with_its_own_tables() -> None:
    doc = _FakeDoc(
        pages={1: object(), 2: object()},
        tables=[_FakeTable(1, "t1"), _FakeTable(2, "t2a"), _FakeTable(2, "t2b")],
    )

    pages = list(document_to_pages(doc))  # type: ignore[arg-type]

    assert [p.page_no for p in pages] == [1, 2]
    assert pages[0].text == "page-1"
    assert pages[0].tables_markdown == ["t1"]
    assert pages[1].tables_markdown == ["t2a", "t2b"]


def test_mapping_collapses_a_pageless_document_to_one_page() -> None:
    doc = _FakeDoc(pages={}, tables=[_FakeTable(1, "only")])

    pages = list(document_to_pages(doc))  # type: ignore[arg-type]

    assert len(pages) == 1
    assert pages[0].page_no == 1
    assert pages[0].text == "whole"
    assert pages[0].tables_markdown == ["only"]


# --- live HTML conversion (structural pipeline, no model downloads) -----------

_HTML = (
    b"<html><body><h1>Risk Factors</h1><p>Markets are volatile.</p>"
    b"<table><tr><th>Year</th><th>Rev</th></tr><tr><td>2023</td><td>100</td></tr></table>"
    b"</body></html>"
)


def test_docling_parses_html_end_to_end() -> None:
    parser = build_docling_parser()
    doc = RawDocument(doc_id="f", filename="f.html", content_type="text/html", data=_HTML)

    pages = list(parser.parse(doc))

    assert len(pages) == 1
    assert "Risk Factors" in pages[0].text
    assert "Markets are volatile." in pages[0].text
    # The table is captured on the structured path.
    assert any("Year" in table and "Rev" in table for table in pages[0].tables_markdown)


def test_docling_parser_is_built_with_a_converter() -> None:
    assert isinstance(build_docling_parser(), DoclingParser)
