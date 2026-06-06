"""Parse router: a detected document in, the right handler out.

The pipeline forks here on a document's detected format (from its bytes, via the
:class:`~raglearn.core.interfaces.FormatDetector`), down one of two arms:

- **Pages** — PDF and IMAGE take Docling's vision pipeline, HTML its structural
  one, plain text a direct decode; anything unrecognized is quarantined. These
  become :class:`~raglearn.core.types.ParsedPage`s.
- **Facts** — XBRL carries the filer's exact figures, so it is extracted as
  :class:`~raglearn.core.types.FinancialFact`s instead of parsed into pages.
  Inline XBRL detects as HTML, so the HTML arm forks on the ``ix:`` namespace; a
  standalone instance and the taxonomy schema (``.xsd``) detect as XML. Facts
  need the instance and its schema, so either one triggers the facts arm (one
  may complete the bundle). Lone linkbases (also XML) are not needed for facts
  and are skipped, pulled in with their instance when it is processed.

The mapping is inherent to the formats, not a tunable, so the router is plain
wiring. Only the detector is config-selected; :func:`build_router` resolves it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping

from raglearn.core.config import Settings
from raglearn.core.interfaces.ingestion import (
    DocumentParser,
    FormatDetector,
    StructuredExtractor,
)
from raglearn.core.types import DetectedFormat, ParsedPage, RawDocument
from raglearn.core.wiring import resolve_adapter
from raglearn.ingestion.bundle import BundleAssembler
from raglearn.ingestion.docling_backend import build_docling_parser
from raglearn.ingestion.object_store import ObjectStore
from raglearn.ingestion.parsers import QuarantineParser, TextParser
from raglearn.ingestion.xbrl_extract import (
    ArelleXbrlExtractor,
    is_inline_xbrl,
    is_xbrl_instance,
    is_xbrl_schema,
)

logger = logging.getLogger(__name__)


class ParserRouter:
    """Routes a document to the pages arm or the facts arm by its format."""

    def __init__(
        self,
        detector: FormatDetector,
        parsers: Mapping[DetectedFormat, DocumentParser],
        extractor: StructuredExtractor,
    ) -> None:
        """Bind the router to a detector, a page-parser table, and a fact extractor.

        Args:
          detector: Identifies a document's format from its bytes.
          parsers: Page parser for each page-producing format; must include an
            :attr:`~raglearn.core.types.DetectedFormat.UNKNOWN` entry, used as
            the fallback for any format absent from the table.
          extractor: Pulls structured facts from XBRL documents.
        """
        if DetectedFormat.UNKNOWN not in parsers:
            raise ValueError("parser table must include an UNKNOWN (quarantine) entry")
        self._detector = detector
        self._parsers = dict(parsers)
        self._extractor = extractor

    def parser_for(self, fmt: DetectedFormat) -> DocumentParser:
        """Return the page parser for a format, falling back to quarantine."""
        return self._parsers.get(fmt, self._parsers[DetectedFormat.UNKNOWN])

    def route(self, document: RawDocument) -> Iterator[ParsedPage]:
        """Detect the document's format and run the page arm.

        For use on page-producing formats; the facts arm and bundle-member
        skipping are decided in :meth:`process`.

        Args:
          document: The fetched document to parse.

        Returns:
          The pages produced by the chosen parser.
        """
        fmt = self._detector.detect(document.data or b"")
        parser = self.parser_for(fmt)
        logger.info("routed %s: %s -> %s", document.doc_id, fmt.value, type(parser).__name__)
        return parser.parse(document)

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
        """Route a document down the facts or pages arm and drain its output.

        Chunking and the structured store attach downstream; until then the
        output is consumed here so the handler actually runs.

        Args:
          document: The fetched document to route.
        """
        data = document.data or b""
        fmt = self._detector.detect(data)
        if self.is_facts_arm(fmt, data):
            facts = sum(1 for _ in self._extractor.extract(document))
            logger.info("routed %s: %s -> XBRL facts (%d)", document.doc_id, fmt.value, facts)
        elif self.is_bundle_member(fmt, data):
            logger.debug("skipped %s: XBRL bundle member", document.doc_id)
        else:
            parser = self.parser_for(fmt)
            pages = sum(1 for _ in parser.parse(document))
            logger.info(
                "routed %s: %s -> %s (%d page(s))",
                document.doc_id,
                fmt.value,
                type(parser).__name__,
                pages,
            )


def default_parsers() -> dict[DetectedFormat, DocumentParser]:
    """Return the format -> page-parser table the pipeline ships with.

    PDF, IMAGE, and HTML share one :class:`~raglearn.ingestion.docling_backend.DoclingParser`:
    Docling's converter routes each to the right internal pipeline (vision for
    PDF/IMAGE, structural for HTML), so they need a single parser instance. XML
    is absent by design — it travels the facts arm, not this table.
    """
    docling = build_docling_parser()
    return {
        DetectedFormat.PDF: docling,
        DetectedFormat.IMAGE: docling,
        DetectedFormat.HTML: docling,
        DetectedFormat.TEXT: TextParser(),
        DetectedFormat.UNKNOWN: QuarantineParser(),
    }


def build_router(settings: Settings, store: ObjectStore) -> ParserRouter:
    """Build a router with the config-active detector, parsers, and XBRL extractor.

    Args:
      settings: Loaded application settings (selects the detector, carries its
        Tika URL).
      store: Object store the XBRL lane reads bundle files from.

    Returns:
      A :class:`ParserRouter` ready to route fetched documents.
    """
    detector = resolve_adapter(settings, "format_detector", url=settings.services.tika_url)
    extractor = ArelleXbrlExtractor(BundleAssembler(store))
    return ParserRouter(detector, default_parsers(), extractor)
