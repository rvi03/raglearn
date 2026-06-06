"""Tests for the per-format page parsers (TextParser, QuarantineParser)."""

from __future__ import annotations

from raglearn.core.types import RawDocument
from raglearn.ingestion.parsers import QuarantineParser, TextParser


def _doc(data: bytes) -> RawDocument:
    return RawDocument(doc_id="d1", filename="d1", content_type="text/plain", data=data)


def test_text_parser_yields_one_page_without_form_feeds() -> None:
    pages = list(TextParser().parse(_doc(b"hello world")))

    assert len(pages) == 1
    assert pages[0].page_no == 1
    assert pages[0].text == "hello world"


def test_text_parser_splits_pages_on_form_feed() -> None:
    pages = list(TextParser().parse(_doc(b"page one\fpage two\fpage three")))

    assert [p.page_no for p in pages] == [1, 2, 3]
    assert [p.text for p in pages] == ["page one", "page two", "page three"]


def test_text_parser_drops_blank_pages_but_keeps_numbering() -> None:
    # A repeated form feed leaves a blank middle page; its number is consumed.
    pages = list(TextParser().parse(_doc(b"first\f\fthird")))

    assert [(p.page_no, p.text) for p in pages] == [(1, "first"), (3, "third")]


def test_text_parser_falls_back_to_latin1_on_invalid_utf8() -> None:
    # 0xE9 is 'é' in Latin-1 but an invalid lone byte in UTF-8.
    pages = list(TextParser().parse(_doc(b"caf\xe9")))

    assert pages[0].text == "café"


def test_text_parser_decodes_utf8() -> None:
    pages = list(TextParser().parse(_doc("café".encode())))

    assert pages[0].text == "café"


def test_text_parser_yields_nothing_for_empty_or_blank() -> None:
    assert list(TextParser().parse(_doc(b""))) == []
    assert list(TextParser().parse(_doc(b"   \n\f\f"))) == []


def test_quarantine_yields_no_pages() -> None:
    assert list(QuarantineParser().parse(_doc(b"anything"))) == []
