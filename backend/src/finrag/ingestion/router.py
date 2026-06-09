"""Parse router: a detected document in, the right arm out.

The pipeline forks here on a document's detected format (from its bytes, via the
:class:`~finrag.core.interfaces.FormatDetector`), down one of two arms:

- **Pages** — narrative documents (PDF/IMAGE via Docling vision, HTML structural,
  text decoded) are parsed to a :class:`~finrag.core.types.ParsedStructure`,
  given a content-based identity, chunked by the adaptive chunk router, embedded,
  and upserted to the vector store. Unrecognized formats are quarantined.
- **Facts** — XBRL carries the filer's exact figures, so it is extracted as
  :class:`~finrag.core.types.FinancialFact`s and written to the structured
  store. Inline XBRL detects as HTML, so the HTML arm forks on the ``ix:``
  namespace; a standalone instance and the taxonomy schema (``.xsd``) detect as
  XML and either triggers extraction (one may complete the bundle). Lone
  linkbases (also XML) are skipped, pulled in with their instance.

Only the format fork is inherent wiring; the adapters it drives are
config-selected and assembled by :func:`build_router`.
"""

from __future__ import annotations

import logging
import time

from finrag.core.config import Settings
from finrag.core.interfaces.crosscutting import MonitorEmitter
from finrag.core.interfaces.ingestion import Embedder, FormatDetector, StructuredExtractor
from finrag.core.interfaces.storage import (
    DedupStore,
    QuarantineStore,
    StructuredStore,
    VectorStore,
)
from finrag.core.types import DetectedFormat, DocumentMetadata, RawDocument
from finrag.core.wiring import build_structured_store, resolve_adapter
from finrag.ingestion.bundle import BundleAssembler
from finrag.ingestion.chunking.parse import StructureParser
from finrag.ingestion.chunking.router import ChunkRouter, build_chunk_router
from finrag.ingestion.chunking.tokenizer import bge_m3_token_counter
from finrag.ingestion.identity import india, metadata_from_identity, resolve_identity_extractor
from finrag.ingestion.identity.llm_classifier import build_doctype_classifier
from finrag.ingestion.identity.rules import content_hash
from finrag.ingestion.identity.us import identity_from_extraction
from finrag.ingestion.monitor import STAGE_LABELS, split_upload_key
from finrag.ingestion.object_store import ObjectStore
from finrag.ingestion.xbrl_extract import (
    ArelleXbrlExtractor,
    is_inline_xbrl,
    is_xbrl_instance,
    is_xbrl_schema,
)
from finrag.observability import NullMonitorEmitter

logger = logging.getLogger(__name__)


def _ms(t0: float) -> str:
    """Elapsed since ``t0`` (a ``perf_counter`` reading), formatted for the trace."""
    return f"{(time.perf_counter() - t0) * 1000:.0f}ms"


def _parse_detail(d: dict[str, object]) -> str:
    """Render the parse decision: lane, shape, and the pdfium-rescue verdict."""
    parts = [
        str(d.get("lane", "?")),
        f"{d.get('sections', 0)} sections",
        f"{d.get('chars', 0):,} chars",
    ]
    if d.get("tables"):
        parts.append(f"{d['tables']} tables")
    if d.get("rescued"):
        dc, pc = d.get("docling_chars", 0), d.get("pdfium_chars", 0)
        parts.append(f"rescued: docling {dc:,} vs pdfium {pc:,} chars")
    return " · ".join(parts)


def _identify_detail(m: DocumentMetadata, extractor: object) -> str:
    """Render the resolved identity: type, company, market, period, and who decided."""
    parts = [m.filing_type, m.company_name, m.market.value]
    if m.fiscal_year:
        parts.append(
            f"FY{m.fiscal_year}-{m.fiscal_period}" if m.fiscal_period else f"FY{m.fiscal_year}"
        )
    if m.ticker:
        parts.append(m.ticker)
    parts.append(f"via {type(extractor).__name__}")
    return " · ".join(parts)


def _chunk_detail(decision: dict[str, object], n: int) -> str:
    """Render the chunk router's decision: winner + each strategy's count/score."""
    strategy = decision.get("strategy") or "?"
    if decision.get("pinned"):
        return f"{strategy} (pinned) · {n} chunks"
    scores = decision.get("scores") or {}
    counts = decision.get("counts") or {}
    if isinstance(scores, dict) and scores and isinstance(counts, dict):
        ranked = " · ".join(
            f"{name} {counts.get(name, '?')}ch @{score}"
            for name, score in sorted(scores.items(), key=lambda kv: -kv[1])
        )
        return f"{strategy} won · {n} chunks  ({ranked})"
    return f"{strategy} · {n} chunks"


class ParserRouter:
    """Routes a document to the pages arm or the facts arm by its format."""

    def __init__(
        self,
        detector: FormatDetector,
        extractor: StructuredExtractor,
        sink: StructuredStore,
        structure_parser: StructureParser,
        chunk_router: ChunkRouter,
        embedder: Embedder,
        vector_store: VectorStore,
        *,
        dedup: DedupStore | None = None,
        quarantine: QuarantineStore | None = None,
        classifier: india.Classifier | None = None,
        emitter: MonitorEmitter | None = None,
    ) -> None:
        """Bind the router to both arms' collaborators.

        Args:
          detector: Identifies a document's format from its bytes.
          extractor: Pulls structured facts from XBRL documents (facts arm).
          sink: Persists extracted facts to the structured store (facts arm).
          structure_parser: Parses a page document into section structure.
          chunk_router: Selects and produces the best chunking for a document.
          embedder: Produces hybrid embeddings for chunks.
          vector_store: Persists embedded chunks (pages arm).
          dedup: Optional content-hash dedup store; when set, the pages arm skips
            documents whose bytes were already ingested.
          quarantine: Optional store for unrecognized-format documents; when set,
            they are recorded instead of only logged.
          classifier: Optional LLM doc-type classifier, injected into the India
            identity extractor as its rules fallback.
          emitter: Optional monitor emitter; each stage transition and terminal
            outcome is published through it for the live DAG. Defaults to a no-op,
            so ingestion runs unobserved unless one is wired in.
        """
        self._detector = detector
        self._extractor = extractor
        self._sink = sink
        self._structure_parser = structure_parser
        self._chunk_router = chunk_router
        self._embedder = embedder
        self._vector_store = vector_store
        self._dedup = dedup
        self._quarantine = quarantine
        self._classifier = classifier
        self._emitter: MonitorEmitter = emitter or NullMonitorEmitter()

    def _node(
        self, document: RawDocument, stage: str, status: str, detail: str | None = None
    ) -> None:
        """Emit one stage transition for a document, if it is a monitored upload.

        The upload key (``<country>/<upload_id>/...``) carries the ids the monitor
        groups by; a document whose id does not parse to one is simply not emitted.
        """
        key = split_upload_key(document.doc_id)
        if key is None:
            return
        self._emitter.node(
            upload_id=key[1],
            doc_id=document.doc_id,
            stage=stage,
            label=STAGE_LABELS[stage],
            status=status,
            detail=detail,
        )

    def _doc_done(self, document: RawDocument, outcome: str) -> None:
        """Emit a document's terminal outcome, if it is a monitored upload."""
        key = split_upload_key(document.doc_id)
        if key is None:
            return
        self._emitter.doc_done(upload_id=key[1], doc_id=document.doc_id, outcome=outcome)

    def is_facts_arm(self, fmt: DetectedFormat, data: bytes) -> bool:
        """Return whether a document goes to the XBRL facts arm.

        The instance and the schema are the two files facts need, so either one
        arriving triggers a completeness check (it may complete the bundle).
        """
        if fmt is DetectedFormat.HTML:
            return is_inline_xbrl(data)
        if fmt is DetectedFormat.XML:
            return is_xbrl_instance(data) or is_xbrl_schema(data)
        return False

    def is_bundle_member(self, fmt: DetectedFormat, data: bytes) -> bool:
        """Return whether a document is a skippable XBRL bundle member.

        XML that is neither an instance nor a schema is a linkbase: not needed
        for facts and pulled in alongside its instance, so on its own it is
        skipped.
        """
        return fmt is DetectedFormat.XML and not is_xbrl_instance(data) and not is_xbrl_schema(data)

    def process(self, document: RawDocument) -> None:
        """Route a document down the facts or pages arm.

        Args:
          document: The fetched document to route.
        """
        data = document.data or b""
        self._node(document, "detect", "running")
        t = time.perf_counter()
        fmt = self._detector.detect(data)
        self._node(
            document, "detect", "done", detail=f"{fmt.value} · {len(data):,} bytes · {_ms(t)}"
        )
        # Route: which arm, and why — the first branch decision.
        self._node(document, "route", "running")
        if self.is_facts_arm(fmt, data):
            why = "inline ix: namespace" if fmt is DetectedFormat.HTML else "XBRL instance/schema"
            self._node(document, "route", "done", detail=f"facts arm · {fmt.value} ({why})")
            self._extract_facts(document, fmt)
        elif self.is_bundle_member(fmt, data):
            self._node(
                document, "route", "skipped", detail="XBRL bundle member (pulled in with instance)"
            )
            self._doc_done(document, "bundled")
            logger.debug("skipped %s: XBRL bundle member", document.doc_id)
        elif fmt is DetectedFormat.UNKNOWN:
            self._node(document, "route", "failed", detail="unrecognized format → quarantine")
            if self._quarantine is not None:
                self._quarantine.quarantine(document.doc_id, fmt.value, "unrecognized format")
            self._doc_done(document, "quarantined")
            logger.warning("quarantined %s: unrecognized format", document.doc_id)
        else:
            self._node(
                document, "route", "done", detail=f"pages arm · {fmt.value} (not inline-XBRL)"
            )
            self._index_pages(document, fmt)

    def _extract_facts(self, document: RawDocument, fmt: DetectedFormat) -> None:
        """Extract a filing's facts and persist them; defer if the bundle is incomplete."""
        self._node(document, "bundle", "running")
        t = time.perf_counter()
        extraction = self._extractor.extract(document)
        if extraction is None:
            self._node(
                document,
                "bundle",
                "deferred",
                detail=f"bundle incomplete (await siblings) · {_ms(t)}",
            )
            self._doc_done(document, "deferred")
            logger.debug("deferred %s: XBRL bundle incomplete", document.doc_id)
            return
        self._node(document, "bundle", "done", detail=f"bundle complete · {_ms(t)}")
        # Stamp the versioning group key (company x period x form family) so
        # restatements can be grouped/superseded later.
        self._node(document, "extract", "running")
        t = time.perf_counter()
        extraction.filing.logical_key = identity_from_extraction(extraction, document).logical_key
        nfacts = len(extraction.facts)
        self._node(
            document,
            "extract",
            "done",
            detail=f"{extraction.filing.filing_id} · {nfacts} facts (Arelle) · {_ms(t)}",
        )
        self._node(document, "write", "running")
        t = time.perf_counter()
        written = self._sink.write(extraction)
        self._node(
            document,
            "write",
            "done",
            detail=f"{written} facts → {type(self._sink).__name__} · {_ms(t)}",
        )
        self._doc_done(document, "facts-written")
        logger.info(
            "routed %s: %s -> XBRL filing %s (%d facts)",
            document.doc_id,
            fmt.value,
            extraction.filing.filing_id,
            written,
        )

    def _index_pages(self, document: RawDocument, fmt: DetectedFormat) -> None:
        """Parse, identify, chunk, embed, and upsert a narrative document.

        Skips the work entirely when the document's bytes were already ingested
        (content-hash dedup), since parsing + embedding are the expensive steps.
        """
        digest = content_hash(document.data or b"")
        if self._dedup is not None and self._dedup.is_ingested(digest):
            self._doc_done(document, "duplicate")
            logger.info("skipped %s: already ingested (content-hash dedup)", document.doc_id)
            return
        self._node(document, "parse", "running")
        t = time.perf_counter()
        structure, pdec = self._structure_parser.parse_with_decision(document, fmt)
        self._node(document, "parse", "done", detail=f"{_parse_detail(pdec)} · {_ms(t)}")
        self._node(document, "identify", "running")
        t = time.perf_counter()
        extractor = resolve_identity_extractor(document.doc_id, classifier=self._classifier)
        identity = extractor.identify(document, structure=structure)
        metadata = metadata_from_identity(identity, document.doc_id)
        self._node(
            document,
            "identify",
            "done",
            detail=f"{_identify_detail(metadata, extractor)} · {_ms(t)}",
        )
        self._node(document, "chunk", "running")
        t = time.perf_counter()
        # Surface the adaptive chunker's actual decision (which strategy won + scores).
        chunks, decision = self._chunk_router.chunk_with_decision(structure, metadata)
        if not chunks:
            self._node(
                document,
                "chunk",
                "done",
                detail=f"0 chunks (no strategy produced output) · {_ms(t)}",
            )
            self._doc_done(document, "empty")
            logger.info("no chunks produced for %s", document.doc_id)
            return
        self._node(
            document, "chunk", "done", detail=f"{_chunk_detail(decision, len(chunks))} · {_ms(t)}"
        )
        self._node(document, "embed", "running")
        t = time.perf_counter()
        vectors = self._embedder.embed([chunk.text for chunk in chunks])
        dims = len(vectors[0].dense) if vectors else 0
        hybrid = "+sparse" if vectors and vectors[0].sparse else ""
        self._node(
            document,
            "embed",
            "done",
            detail=f"bge-m3 · {dims}-d dense{hybrid} · {len(vectors)} vectors · {_ms(t)}",
        )
        self._node(document, "index", "running")
        t = time.perf_counter()
        self._vector_store.upsert(chunks, vectors)
        target = getattr(self._vector_store, "_collection", type(self._vector_store).__name__)
        self._node(document, "index", "done", detail=f"+{len(chunks)} points → {target} · {_ms(t)}")
        if self._dedup is not None:
            self._dedup.mark_ingested(digest, document.doc_id)
        self._doc_done(document, "indexed")
        logger.info("indexed %s: %s -> %d chunk(s)", document.doc_id, fmt.value, len(chunks))


def build_router(
    settings: Settings, store: ObjectStore, *, emitter: MonitorEmitter | None = None
) -> ParserRouter:
    """Assemble a router from config: both arms' adapters and the chunk pipeline.

    Args:
      settings: Loaded settings (selects detector, structured store, embedder,
        vector store, LLM backend; carries service URLs and the DuckDB path).
      store: Object store the XBRL lane reads bundle files from.
      emitter: Optional monitor emitter for the live ingestion DAG; defaults to a
        no-op, so the router stays silent unless the consumer wires one in.

    Returns:
      A :class:`ParserRouter` ready to route fetched documents.
    """
    detector = resolve_adapter(settings, "format_detector", url=settings.services.tika_url)
    extractor = ArelleXbrlExtractor(BundleAssembler(store))
    sink = build_structured_store(settings)
    embedder = resolve_adapter(settings, "embedder", batch_size=settings.ingestion.embed_batch_size)
    vector_store = resolve_adapter(settings, "vector_store", url=settings.services.qdrant_url)
    chunk_router = build_chunk_router(embedder, bge_m3_token_counter())
    classifier = build_doctype_classifier(settings)
    return ParserRouter(
        detector,
        extractor,
        sink,
        StructureParser(),
        chunk_router,
        embedder,
        vector_store,
        dedup=sink,  # the DuckDB store is also the content-hash dedup + quarantine store
        quarantine=sink,
        classifier=classifier,
        emitter=emitter,
    )
