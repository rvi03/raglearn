"""Tests for the parse router's two-arm dispatch (pages vs. XBRL facts).

Hermetic: a fake detector returns a fixed format, recording handlers capture
which arm ran, and real classifier markers (inline-XBRL / instance / linkbase
namespaces) drive the fork — no Tika, parser backend, or object store needed.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from raglearn.core.types import (
    CollectionMetadata,
    DetectedFormat,
    FactOrigin,
    FilingMetadata,
    FinancialFact,
    Market,
    ParsedPage,
    RawDocument,
    XbrlExtraction,
)
from raglearn.ingestion.docling_backend import DoclingParser
from raglearn.ingestion.parsers import QuarantineParser, TextParser
from raglearn.ingestion.router import ParserRouter, default_parsers

# Byte markers the real classifier keys on.
_IXBRL = b'<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><body>'
_INSTANCE = b'<xbrl xmlns="http://www.xbrl.org/2003/instance">'
_LINKBASE = b'<linkbase xmlns="http://www.xbrl.org/2003/linkbase">'
_SCHEMA = (
    b'<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema" '
    b'xmlns:xbrli="http://www.xbrl.org/2003/instance">'
)


class _FakeDetector:
    """Detector that always reports a preset format and records what it saw."""

    def __init__(self, fmt: DetectedFormat) -> None:
        self.fmt = fmt
        self.seen: bytes | None = None

    def detect(self, data: bytes) -> DetectedFormat:
        self.seen = data
        return self.fmt


class _RecordingParser:
    """Page parser that records the documents handed to it and yields no pages."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def parse(self, document: RawDocument) -> Iterator[ParsedPage]:
        self.calls.append(document.doc_id)
        return iter(())


def _extraction(facts: list[FinancialFact] | None = None) -> XbrlExtraction:
    return XbrlExtraction(
        collection=CollectionMetadata(collection_id="us-cik-1", market=Market.US),
        filing=FilingMetadata(filing_id="acc-1", collection_id="us-cik-1"),
        facts=facts or [],
    )


class _RecordingExtractor:
    """Extractor returning a preset extraction (or None to defer), recording calls."""

    def __init__(self, extraction: XbrlExtraction | None = None, *, defer: bool = False) -> None:
        self.calls: list[str] = []
        self._extraction = extraction if extraction is not None else _extraction()
        self._defer = defer

    def extract(self, document: RawDocument) -> XbrlExtraction | None:
        self.calls.append(document.doc_id)
        return None if self._defer else self._extraction


class _RecordingSink:
    """Structured sink that captures every extraction written to it."""

    def __init__(self) -> None:
        self.writes: list[XbrlExtraction] = []

    def write(self, extraction: XbrlExtraction) -> int:
        self.writes.append(extraction)
        return len(extraction.facts)


def _doc(data: bytes = b"bytes") -> RawDocument:
    return RawDocument(doc_id="d1", filename="d1", content_type="x", data=data)


def _router(
    fmt: DetectedFormat,
    parsers: dict[DetectedFormat, _RecordingParser] | None = None,
    extractor: _RecordingExtractor | None = None,
    sink: _RecordingSink | None = None,
) -> ParserRouter:
    table = parsers if parsers is not None else {DetectedFormat.UNKNOWN: _RecordingParser()}
    return ParserRouter(
        _FakeDetector(fmt), table, extractor or _RecordingExtractor(), sink or _RecordingSink()
    )


# --- pages arm ----------------------------------------------------------------


def test_route_dispatches_to_the_detected_formats_parser() -> None:
    pdf_parser = _RecordingParser()
    router = _router(
        DetectedFormat.PDF,
        {DetectedFormat.PDF: pdf_parser, DetectedFormat.UNKNOWN: _RecordingParser()},
    )

    list(router.route(_doc()))

    assert pdf_parser.calls == ["d1"]


def test_route_passes_document_bytes_to_the_detector() -> None:
    detector = _FakeDetector(DetectedFormat.TEXT)
    router = ParserRouter(
        detector,
        {DetectedFormat.UNKNOWN: _RecordingParser()},
        _RecordingExtractor(),
        _RecordingSink(),
    )

    list(router.route(_doc(b"hello")))

    assert detector.seen == b"hello"


def test_unmapped_format_falls_back_to_quarantine() -> None:
    quarantine = _RecordingParser()
    router = _router(
        DetectedFormat.IMAGE,  # not in the table below
        {DetectedFormat.PDF: _RecordingParser(), DetectedFormat.UNKNOWN: quarantine},
    )

    list(router.route(_doc()))

    assert quarantine.calls == ["d1"]


def test_table_without_unknown_entry_is_rejected() -> None:
    with pytest.raises(ValueError, match="UNKNOWN"):
        ParserRouter(
            _FakeDetector(DetectedFormat.PDF),
            {DetectedFormat.PDF: _RecordingParser()},
            _RecordingExtractor(),
            _RecordingSink(),
        )


def test_process_runs_the_page_parser_and_drains() -> None:
    parser = _RecordingParser()
    router = _router(
        DetectedFormat.PDF,
        {DetectedFormat.PDF: parser, DetectedFormat.UNKNOWN: _RecordingParser()},
    )

    router.process(_doc())

    assert parser.calls == ["d1"]


# --- facts arm + bundle-member skipping --------------------------------------


def test_inline_xbrl_html_goes_to_the_facts_arm() -> None:
    parser, extractor = _RecordingParser(), _RecordingExtractor()
    router = _router(
        DetectedFormat.HTML,
        {DetectedFormat.HTML: parser, DetectedFormat.UNKNOWN: parser},
        extractor,
    )

    router.process(_doc(_IXBRL))

    assert extractor.calls == ["d1"]
    assert parser.calls == []


def test_plain_html_goes_to_the_pages_arm() -> None:
    parser, extractor = _RecordingParser(), _RecordingExtractor()
    router = _router(
        DetectedFormat.HTML,
        {DetectedFormat.HTML: parser, DetectedFormat.UNKNOWN: parser},
        extractor,
    )

    router.process(_doc(b"<html><body>plain</body></html>"))

    assert parser.calls == ["d1"]
    assert extractor.calls == []


def test_facts_arm_writes_the_extraction_to_the_sink() -> None:
    fact = FinancialFact(
        fact_id="f1",
        filing_id="acc-1",
        concept="mock:Revenue",
        value=100.0,
        unit="USD",
        period="2024-01-01/2024-12-31",
        origin=FactOrigin.XBRL,
    )
    extraction = _extraction([fact])
    extractor, sink = _RecordingExtractor(extraction), _RecordingSink()
    router = _router(
        DetectedFormat.XML, {DetectedFormat.UNKNOWN: _RecordingParser()}, extractor, sink
    )

    router.process(_doc(_INSTANCE))

    assert sink.writes == [extraction]
    assert sink.writes[0].facts == [fact]


def test_facts_arm_with_an_incomplete_bundle_writes_nothing() -> None:
    extractor, sink = _RecordingExtractor(defer=True), _RecordingSink()
    router = _router(
        DetectedFormat.XML, {DetectedFormat.UNKNOWN: _RecordingParser()}, extractor, sink
    )

    router.process(_doc(_INSTANCE))

    assert extractor.calls == ["d1"]  # extraction attempted
    assert sink.writes == []  # but nothing persisted (deferred)


def test_standalone_xbrl_instance_goes_to_the_facts_arm() -> None:
    parser, extractor = _RecordingParser(), _RecordingExtractor()
    router = _router(DetectedFormat.XML, {DetectedFormat.UNKNOWN: parser}, extractor)

    router.process(_doc(_INSTANCE))

    assert extractor.calls == ["d1"]


def test_xbrl_schema_goes_to_the_facts_arm() -> None:
    # A schema event may be the file that completes the bundle, so it triggers.
    parser, extractor = _RecordingParser(), _RecordingExtractor()
    router = _router(DetectedFormat.XML, {DetectedFormat.UNKNOWN: parser}, extractor)

    router.process(_doc(_SCHEMA))

    assert extractor.calls == ["d1"]


def test_xbrl_linkbase_is_skipped_as_a_bundle_member() -> None:
    parser, extractor = _RecordingParser(), _RecordingExtractor()
    router = _router(DetectedFormat.XML, {DetectedFormat.UNKNOWN: parser}, extractor)

    router.process(_doc(_LINKBASE))

    assert extractor.calls == []
    assert parser.calls == []  # neither extracted nor parsed


# --- default table ------------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "parser_type"),
    [
        (DetectedFormat.PDF, DoclingParser),
        (DetectedFormat.IMAGE, DoclingParser),
        (DetectedFormat.HTML, DoclingParser),
        (DetectedFormat.TEXT, TextParser),
        (DetectedFormat.UNKNOWN, QuarantineParser),
    ],
)
def test_default_table_maps_each_format_to_its_parser(
    fmt: DetectedFormat, parser_type: type
) -> None:
    assert isinstance(default_parsers()[fmt], parser_type)


def test_default_table_shares_one_docling_parser_for_pdf_image_html() -> None:
    table = default_parsers()
    assert table[DetectedFormat.PDF] is table[DetectedFormat.IMAGE]
    assert table[DetectedFormat.IMAGE] is table[DetectedFormat.HTML]


def test_default_table_excludes_xml_which_takes_the_facts_arm() -> None:
    assert DetectedFormat.XML not in default_parsers()
    assert set(default_parsers()) == set(DetectedFormat) - {DetectedFormat.XML}
