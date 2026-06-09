"""Tests for the parse router's two-arm dispatch (pages vs. XBRL facts).

Hermetic: a fake detector returns a fixed format; the pages arm runs against fake
structure-parser / chunk-router / embedder / vector-store collaborators (real
identity adapters, keyed off the country path prefix); the facts arm uses
recording extractor/sink. Real classifier markers drive the fork. No Tika,
Docling, Arelle, Qdrant, or object store needed.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from finrag.core.types import (
    BlockKind,
    Chunk,
    ChunkType,
    CollectionMetadata,
    DetectedFormat,
    DocumentMetadata,
    EmbeddingVector,
    FactOrigin,
    FilingMetadata,
    FinancialFact,
    Market,
    ParsedStructure,
    RawDocument,
    StructureBlock,
    StructureSection,
    XbrlExtraction,
)
from finrag.ingestion.router import ParserRouter

# Byte markers the real XBRL classifier keys on.
_IXBRL = b'<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><body>'
_INSTANCE = b'<xbrl xmlns="http://www.xbrl.org/2003/instance">'
_LINKBASE = b'<linkbase xmlns="http://www.xbrl.org/2003/linkbase">'
_SCHEMA = (
    b'<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema" '
    b'xmlns:xbrli="http://www.xbrl.org/2003/instance">'
)
_INDIA_KEY = "india/mockpharma/Q4FY2026/results.pdf"


class _FakeDetector:
    def __init__(self, fmt: DetectedFormat) -> None:
        self.fmt = fmt
        self.seen: bytes | None = None

    def detect(self, data: bytes) -> DetectedFormat:
        self.seen = data
        return self.fmt


class _RecordingExtractor:
    def __init__(self, extraction: XbrlExtraction | None = None, *, defer: bool = False) -> None:
        self.calls: list[str] = []
        self._extraction = extraction if extraction is not None else _extraction()
        self._defer = defer

    def extract(self, document: RawDocument) -> XbrlExtraction | None:
        self.calls.append(document.doc_id)
        return None if self._defer else self._extraction


class _RecordingSink:
    def __init__(self) -> None:
        self.writes: list[XbrlExtraction] = []

    def write(self, extraction: XbrlExtraction) -> int:
        self.writes.append(extraction)
        return len(extraction.facts)


class _FakeStructureParser:
    def __init__(self) -> None:
        self.calls: list[tuple[str, DetectedFormat]] = []

    def parse(self, document: RawDocument, fmt: DetectedFormat) -> ParsedStructure:
        structure, _ = self.parse_with_decision(document, fmt)
        return structure

    def parse_with_decision(
        self, document: RawDocument, fmt: DetectedFormat
    ) -> tuple[ParsedStructure, dict[str, object]]:
        self.calls.append((document.doc_id, fmt))
        block = StructureBlock(kind=BlockKind.TEXT, text="body")
        section = StructureSection(title="Financial Results", level=0, blocks=[block])
        structure = ParsedStructure(source_doc_id=document.doc_id, sections=[section])
        return structure, {"lane": "fake", "sections": 1, "chars": 4, "tables": 0}


class _FakeChunkRouter:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.metadata: list[DocumentMetadata] = []
        self._chunks = chunks

    def chunk(self, structure: ParsedStructure, metadata: DocumentMetadata) -> Iterator[Chunk]:
        self.metadata.append(metadata)
        return iter(self._chunks)

    def chunk_with_decision(
        self, structure: ParsedStructure, metadata: DocumentMetadata
    ) -> tuple[list[Chunk], dict[str, object]]:
        self.metadata.append(metadata)
        return list(self._chunks), {"strategy": "structure_aware", "scores": {}}


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        self.texts = list(texts)
        return [EmbeddingVector(dense=[1.0]) for _ in texts]


class _RecordingVectorStore:
    def __init__(self) -> None:
        self.upserts: list[tuple[list[Chunk], list[EmbeddingVector]]] = []

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[EmbeddingVector]) -> None:
        self.upserts.append((list(chunks), list(vectors)))

    def search(
        self,
        vector: EmbeddingVector,
        *,
        top_k: int,
        filters: dict[str, str],
        access_tags: Sequence[str] = (),
    ) -> list:
        return []


def _extraction(facts: list[FinancialFact] | None = None) -> XbrlExtraction:
    return XbrlExtraction(
        collection=CollectionMetadata(collection_id="us-cik-1", market=Market.US),
        filing=FilingMetadata(filing_id="acc-1", collection_id="us-cik-1"),
        facts=facts or [],
    )


def _chunk() -> Chunk:
    md = DocumentMetadata(
        collection_id="in-mockpharma",
        company_name="mockpharma",
        market=Market.IN,
        filing_type="financial_results",
        source_doc_id=_INDIA_KEY,
        content_hash="h",
        logical_key="lk",
    )
    return Chunk(chunk_id="c0", text="chunk text", chunk_type=ChunkType.TEXT, metadata=md)


def _doc(data: bytes = b"bytes", doc_id: str = _INDIA_KEY) -> RawDocument:
    return RawDocument(doc_id=doc_id, filename="f", content_type="x", data=data)


class _RecordingEmitter:
    """Records the monitor events the router emits, for sequence assertions."""

    def __init__(self) -> None:
        self.nodes: list[tuple[str, str, str | None]] = []  # (stage, status, detail)
        self.done: list[tuple[str, str]] = []  # (upload_id, outcome)

    def upload(self, **kwargs: object) -> None:  # pragma: no cover - router never calls this
        raise AssertionError("router must not emit upload events")

    def node(
        self,
        *,
        upload_id: str,
        doc_id: str,
        stage: str,
        label: str,
        status: str,
        detail: str | None = None,
    ) -> None:
        self.nodes.append((stage, status, detail))

    def doc_done(self, *, upload_id: str, doc_id: str, outcome: str) -> None:
        self.done.append((upload_id, outcome))


def _router(
    fmt: DetectedFormat,
    *,
    extractor: _RecordingExtractor | None = None,
    sink: _RecordingSink | None = None,
    structure_parser: _FakeStructureParser | None = None,
    chunk_router: _FakeChunkRouter | None = None,
    embedder: _RecordingEmbedder | None = None,
    vector_store: _RecordingVectorStore | None = None,
    emitter: _RecordingEmitter | None = None,
) -> ParserRouter:
    return ParserRouter(
        _FakeDetector(fmt),
        extractor or _RecordingExtractor(),
        sink or _RecordingSink(),
        structure_parser or _FakeStructureParser(),
        chunk_router or _FakeChunkRouter([_chunk()]),
        embedder or _RecordingEmbedder(),
        vector_store or _RecordingVectorStore(),
        emitter=emitter,
    )


# --- pages arm ----------------------------------------------------------------


def test_pages_arm_chunks_embeds_and_upserts() -> None:
    chunk_router = _FakeChunkRouter([_chunk()])
    embedder, store = _RecordingEmbedder(), _RecordingVectorStore()
    router = _router(
        DetectedFormat.PDF, chunk_router=chunk_router, embedder=embedder, vector_store=store
    )

    router.process(_doc(b"pdf-bytes"))

    assert len(store.upserts) == 1
    chunks, vectors = store.upserts[0]
    assert [c.chunk_id for c in chunks] == ["c0"]
    assert len(vectors) == 1
    assert embedder.texts == ["chunk text"]
    # identity flowed onto the metadata the chunker received
    assert chunk_router.metadata[0].market is Market.IN
    assert chunk_router.metadata[0].content_hash is not None


class _Dedup:
    def __init__(self, *, seen: bool) -> None:
        self._seen = seen
        self.marked: list[str] = []

    def is_ingested(self, content_hash: str) -> bool:
        return self._seen

    def mark_ingested(self, content_hash: str, doc_id: str) -> None:
        self.marked.append(content_hash)


def test_pages_arm_skips_when_dedup_reports_seen() -> None:
    store = _RecordingVectorStore()
    router = ParserRouter(
        _FakeDetector(DetectedFormat.PDF),
        _RecordingExtractor(),
        _RecordingSink(),
        _FakeStructureParser(),
        _FakeChunkRouter([_chunk()]),
        _RecordingEmbedder(),
        store,
        dedup=_Dedup(seen=True),
    )

    router.process(_doc())

    assert store.upserts == []  # identical bytes already ingested -> skipped


def test_pages_arm_marks_dedup_after_indexing() -> None:
    store, dedup = _RecordingVectorStore(), _Dedup(seen=False)
    router = ParserRouter(
        _FakeDetector(DetectedFormat.PDF),
        _RecordingExtractor(),
        _RecordingSink(),
        _FakeStructureParser(),
        _FakeChunkRouter([_chunk()]),
        _RecordingEmbedder(),
        store,
        dedup=dedup,
    )

    router.process(_doc())

    assert len(store.upserts) == 1
    assert len(dedup.marked) == 1  # recorded as ingested after a successful index


class _RecordingQuarantine:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, str]] = []

    def quarantine(self, doc_id: str, detected_format: str, reason: str) -> None:
        self.records.append((doc_id, detected_format, reason))


def test_unknown_format_is_quarantined() -> None:
    structure_parser, store = _FakeStructureParser(), _RecordingVectorStore()
    extractor, quarantine = _RecordingExtractor(), _RecordingQuarantine()
    router = ParserRouter(
        _FakeDetector(DetectedFormat.UNKNOWN),
        extractor,
        _RecordingSink(),
        structure_parser,
        _FakeChunkRouter([_chunk()]),
        _RecordingEmbedder(),
        store,
        quarantine=quarantine,
    )

    router.process(_doc())

    assert structure_parser.calls == []
    assert store.upserts == []
    assert extractor.calls == []
    assert quarantine.records == [(_INDIA_KEY, "unknown", "unrecognized format")]


def test_pages_arm_skips_upsert_when_no_chunks() -> None:
    store = _RecordingVectorStore()
    router = _router(DetectedFormat.PDF, chunk_router=_FakeChunkRouter([]), vector_store=store)

    router.process(_doc())

    assert store.upserts == []


def test_process_passes_bytes_to_detector() -> None:
    detector = _FakeDetector(DetectedFormat.PDF)
    router = ParserRouter(
        detector,
        _RecordingExtractor(),
        _RecordingSink(),
        _FakeStructureParser(),
        _FakeChunkRouter([_chunk()]),
        _RecordingEmbedder(),
        _RecordingVectorStore(),
    )

    router.process(_doc(b"hello"))

    assert detector.seen == b"hello"


# --- facts arm + bundle-member skipping --------------------------------------


def test_inline_xbrl_html_goes_to_the_facts_arm() -> None:
    extractor, structure_parser = _RecordingExtractor(), _FakeStructureParser()
    router = _router(DetectedFormat.HTML, extractor=extractor, structure_parser=structure_parser)

    router.process(_doc(_IXBRL))

    assert extractor.calls == [_INDIA_KEY]
    assert structure_parser.calls == []  # facts arm, not pages


def test_plain_html_goes_to_the_pages_arm() -> None:
    extractor, structure_parser = _RecordingExtractor(), _FakeStructureParser()
    router = _router(DetectedFormat.HTML, extractor=extractor, structure_parser=structure_parser)

    router.process(_doc(b"<html><body>plain</body></html>"))

    assert structure_parser.calls and structure_parser.calls[0][0] == _INDIA_KEY
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
    router = _router(DetectedFormat.XML, extractor=extractor, sink=sink)

    router.process(_doc(_INSTANCE))

    assert sink.writes == [extraction]


def test_facts_arm_with_an_incomplete_bundle_writes_nothing() -> None:
    extractor, sink = _RecordingExtractor(defer=True), _RecordingSink()
    router = _router(DetectedFormat.XML, extractor=extractor, sink=sink)

    router.process(_doc(_INSTANCE))

    assert extractor.calls == [_INDIA_KEY]
    assert sink.writes == []


def test_standalone_instance_and_schema_go_to_the_facts_arm() -> None:
    for marker in (_INSTANCE, _SCHEMA):
        extractor = _RecordingExtractor()
        _router(DetectedFormat.XML, extractor=extractor).process(_doc(marker))
        assert extractor.calls == [_INDIA_KEY]


def test_xbrl_linkbase_is_skipped_as_a_bundle_member() -> None:
    extractor, structure_parser = _RecordingExtractor(), _FakeStructureParser()
    router = _router(DetectedFormat.XML, extractor=extractor, structure_parser=structure_parser)

    router.process(_doc(_LINKBASE))

    assert extractor.calls == []
    assert structure_parser.calls == []  # neither extracted nor parsed


# --- monitor emission ---------------------------------------------------------


def test_pages_arm_emits_the_full_stage_sequence() -> None:
    emitter = _RecordingEmitter()
    _router(DetectedFormat.PDF, emitter=emitter).process(_doc(b"pdf-bytes"))

    stages = [(stage, status) for stage, status, _ in emitter.nodes]
    assert stages == [
        ("detect", "running"),
        ("detect", "done"),
        ("route", "running"),
        ("route", "done"),
        ("parse", "running"),
        ("parse", "done"),
        ("identify", "running"),
        ("identify", "done"),
        ("chunk", "running"),
        ("chunk", "done"),
        ("embed", "running"),
        ("embed", "done"),
        ("index", "running"),
        ("index", "done"),
    ]
    assert emitter.done == [("mockpharma", "indexed")]


def test_pages_arm_emits_duplicate_outcome_on_dedup_hit() -> None:
    emitter = _RecordingEmitter()
    router = ParserRouter(
        _FakeDetector(DetectedFormat.PDF),
        _RecordingExtractor(),
        _RecordingSink(),
        _FakeStructureParser(),
        _FakeChunkRouter([_chunk()]),
        _RecordingEmbedder(),
        _RecordingVectorStore(),
        dedup=_Dedup(seen=True),
        emitter=emitter,
    )

    router.process(_doc(b"pdf-bytes"))

    # Detect + route run, then the doc short-circuits — no parse/chunk/embed work.
    assert [stage for stage, _, _ in emitter.nodes] == ["detect", "detect", "route", "route"]
    assert emitter.done == [("mockpharma", "duplicate")]


def test_pages_arm_emits_empty_outcome_when_no_chunks() -> None:
    emitter = _RecordingEmitter()
    _router(DetectedFormat.PDF, chunk_router=_FakeChunkRouter([]), emitter=emitter).process(
        _doc(b"pdf-bytes")
    )

    assert any(
        s == "chunk" and st == "done" and "0 chunks" in (d or "") for s, st, d in emitter.nodes
    )
    assert ("embed", "running", None) not in emitter.nodes  # never embedded
    assert emitter.done == [("mockpharma", "empty")]


def test_facts_arm_emits_bundle_extract_write_then_facts() -> None:
    emitter = _RecordingEmitter()
    _router(DetectedFormat.XML, emitter=emitter).process(_doc(_INSTANCE))

    stages = [(stage, status) for stage, status, _ in emitter.nodes]
    assert stages == [
        ("detect", "running"),
        ("detect", "done"),
        ("route", "running"),
        ("route", "done"),
        ("bundle", "running"),
        ("bundle", "done"),
        ("extract", "running"),
        ("extract", "done"),
        ("write", "running"),
        ("write", "done"),
    ]
    assert emitter.done == [("mockpharma", "facts-written")]


def test_facts_arm_emits_deferred_outcome_on_incomplete_bundle() -> None:
    emitter = _RecordingEmitter()
    _router(DetectedFormat.XML, extractor=_RecordingExtractor(defer=True), emitter=emitter).process(
        _doc(_INSTANCE)
    )

    assert any(
        s == "bundle" and st == "deferred" and "incomplete" in (d or "")
        for s, st, d in emitter.nodes
    )
    assert emitter.done == [("mockpharma", "deferred")]


def test_unknown_format_emits_quarantined_outcome() -> None:
    emitter = _RecordingEmitter()
    _router(DetectedFormat.UNKNOWN, emitter=emitter).process(_doc(b"???"))

    assert emitter.done == [("mockpharma", "quarantined")]


def test_bundle_member_emits_bundled_outcome() -> None:
    emitter = _RecordingEmitter()
    _router(DetectedFormat.XML, emitter=emitter).process(_doc(_LINKBASE))

    assert emitter.done == [("mockpharma", "bundled")]


def test_no_emission_when_doc_id_is_not_a_monitored_upload() -> None:
    # A two-segment key has no <country>/<upload_id>/<relpath> shape, so the
    # monitor helpers no-op. Uses the bundle-member path (detect + done only) to
    # avoid identity resolution, which the upload-keyed paths exercise elsewhere.
    emitter = _RecordingEmitter()
    _router(DetectedFormat.XML, emitter=emitter).process(_doc(_LINKBASE, doc_id="us/loose.xml"))

    assert emitter.nodes == []
    assert emitter.done == []
