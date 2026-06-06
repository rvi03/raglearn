"""Tests for grouping a classified item stream into section structure.

Parser-agnostic: exercises ``assemble_sections`` with a synthetic item stream,
so the two parser adapters (SEC HTML, Docling PDF) share these guarantees
without needing network access or real filings.
"""

from __future__ import annotations

from raglearn.core.types import BlockKind
from raglearn.ingestion.chunking.assemble import ParsedItem, assemble_sections


def test_titles_open_sections_and_content_fills_them() -> None:
    items = [
        ParsedItem("title", "Item 1A.", 0),
        ParsedItem("text", "Our business faces risks.", 0),
        ParsedItem("table", "| a | b |", 0),
        ParsedItem("title", "Item 2.", 0),
        ParsedItem("text", "We lease facilities.", 0),
    ]

    structure = assemble_sections(items, "doc-1")

    assert [s.title for s in structure.sections] == ["Item 1A.", "Item 2."]
    first = structure.sections[0]
    assert [b.kind for b in first.blocks] == [BlockKind.TEXT, BlockKind.TABLE]
    assert structure.source_doc_id == "doc-1"


def test_content_before_first_title_becomes_an_untitled_preamble() -> None:
    items = [
        ParsedItem("text", "Cover page.", 0),
        ParsedItem("title", "Item 1.", 0),
        ParsedItem("text", "Business.", 0),
    ]

    structure = assemble_sections(items, "doc-2")

    assert structure.sections[0].title is None
    assert structure.sections[0].blocks[0].text == "Cover page."
    assert structure.sections[1].title == "Item 1."


def test_subtitles_carry_their_nesting_level() -> None:
    items = [
        ParsedItem("title", "Item 1.", 0),
        ParsedItem("subtitle", "Products", 1),
        ParsedItem("text", "We sell devices.", 0),
    ]

    structure = assemble_sections(items, "doc-3")

    assert [(s.title, s.level) for s in structure.sections] == [("Item 1.", 0), ("Products", 1)]


def test_page_is_carried_onto_blocks() -> None:
    items = [
        ParsedItem("title", "Item 7.", 0, page=12),
        ParsedItem("text", "MD&A prose.", 0, page=12),
    ]

    structure = assemble_sections(items, "doc-pdf")

    assert structure.sections[0].blocks[0].page == 12


def test_blank_blocks_and_empty_titles_are_dropped() -> None:
    items = [
        ParsedItem("title", "   ", 0),
        ParsedItem("text", "   ", 0),
        ParsedItem("text", "Real content.", 0),
    ]

    structure = assemble_sections(items, "doc-4")

    assert len(structure.sections) == 1
    assert structure.sections[0].title is None
    assert [b.text for b in structure.sections[0].blocks] == ["Real content."]


def test_empty_item_stream_yields_no_sections() -> None:
    assert assemble_sections([], "doc-5").sections == []
