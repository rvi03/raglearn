"""XBRL fact extraction via Arelle.

US exact figures come from XBRL, not from parsing tables: the filer's as-filed
values, with zero OCR. Arelle (the reference XBRL processor) reads a filing's
inline-XBRL instance together with its schema and linkbases and resolves every
tagged value to its ``us-gaap``/``dei`` concept, period, unit, and dimensions.

Two concerns live here:

- :func:`extract_facts` — given a local instance file (with its bundle siblings
  beside it on disk), return deduplicated numeric :class:`FinancialFact`s.
- classification (:func:`is_inline_xbrl`, :func:`is_xbrl_instance`) — the content
  step that tells the router an instance worth extracting from a bundle member
  (a lone linkbase or schema) that should be skipped, since the bundle is pulled
  whole when its instance is seen.

edgartools is intentionally *not* used for facts: its local inline-XBRL path
does not bind facts to concepts in the version pinned here. Arelle does.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import timedelta
from pathlib import Path

from arelle import Cntlr

from raglearn.core.errors import IngestionError
from raglearn.core.types import (
    CollectionMetadata,
    FactOrigin,
    FilingMetadata,
    FinancialFact,
    Market,
    RawDocument,
    XbrlExtraction,
)
from raglearn.ingestion.bundle import BundleAssembler

logger = logging.getLogger(__name__)

# How many leading bytes to sniff when classifying an XML/HTML document. The
# inline-XBRL namespace and the instance/linkbase root element both appear in the
# document head, so a prefix is enough and keeps large filings off the heap.
_SNIFF_BYTES = 65536

# Markers are stored lowercased to match _sniff, which lowercases the head so the
# checks are case-insensitive (element/attribute case varies across filers).
# Marks the document carries inline XBRL facts (an iXBRL filing detects as HTML).
_INLINE_XBRL_NS = "http://www.xbrl.org/2013/inlinexbrl"

# Roots that mean "this XML is an XBRL instance" vs. a bundle member to skip.
_INSTANCE_NS = "http://www.xbrl.org/2003/instance"
_LINKBASE_NS = "http://www.xbrl.org/2003/linkbase"

# DEI (Document & Entity Information) is the SEC's self-describing filing metadata
# — company, CIK, ticker, document type, fiscal period. Its namespace is versioned
# (e.g. http://xbrl.sec.gov/dei/2024), so we match on the stable ``/dei/`` segment.
_DEI_NS_MARKER = "/dei/"


def _sniff(data: bytes) -> str:
    """Decode the document head for cheap, case-insensitive marker checks."""
    return data[:_SNIFF_BYTES].decode("utf-8", errors="ignore").lower()


def is_inline_xbrl(data: bytes) -> bool:
    """Return whether HTML bytes carry inline XBRL (the ``ix:`` content fork).

    An inline-XBRL filing is xhtml, so it detects as HTML; only the ones that
    declare the inline-XBRL namespace go to the facts path instead of the
    document parser.
    """
    return _INLINE_XBRL_NS in _sniff(data)


def is_xbrl_instance(data: bytes) -> bool:
    """Return whether XML bytes are an XBRL instance, not a linkbase or schema.

    The instance is the only bundle member that triggers extraction; linkbases
    (``_cal``/``_def``/``_lab``/``_pre``) and the schema (``.xsd``) are pulled in
    alongside it, not processed on their own.
    """
    head = _sniff(data)
    if _INSTANCE_NS not in head:
        return False
    # A linkbase also references the instance namespace in places; exclude it by
    # its own root namespace and the schema's root element.
    return _LINKBASE_NS not in head and "<xsd:schema" not in head and "<schema" not in head


def is_xbrl_schema(data: bytes) -> bool:
    """Return whether XML bytes are an XBRL taxonomy schema (the filing's ``.xsd``).

    The schema is a bundle member like the linkbases, but a *required* one — the
    instance's facts cannot be resolved without it — so a schema event triggers a
    completeness check (it may be the file that completes the bundle), whereas a
    linkbase event does not.
    """
    head = _sniff(data)
    is_schema = "<xsd:schema" in head or "<schema" in head
    return is_schema and "xbrl" in head


def _period(context: object) -> str:
    """Render a context's period, correcting Arelle's exclusive end date.

    Arelle stores a period end as midnight of the following day, so a day is
    subtracted to get the conventional, as-filed end date. Start dates are
    inclusive and used as-is.
    """
    ctx = context
    if ctx.isInstantPeriod:  # type: ignore[attr-defined]
        instant = ctx.instantDatetime - timedelta(days=1)  # type: ignore[attr-defined]
        return str(instant.date().isoformat())
    start = ctx.startDatetime.date().isoformat()  # type: ignore[attr-defined]
    end = (ctx.endDatetime - timedelta(days=1)).date().isoformat()  # type: ignore[attr-defined]
    return f"{start}/{end}"


def _unit(fact: object) -> str:
    """Render a fact's unit (e.g. ``USD``, ``shares``, ``USD/shares``)."""
    unit = fact.unit  # type: ignore[attr-defined]
    if unit is None or not unit.measures:
        return ""
    numerators, denominators = unit.measures
    num = "*".join(m.localName for m in numerators)
    den = "*".join(m.localName for m in denominators)
    return f"{num}/{den}" if den else num


def _member(value: object) -> str:
    """Render a dimension value's member: a ``QName`` if explicit, else its typed value.

    XBRL dimensions come in two kinds. An **explicit** member is an enumerated
    ``QName`` from a domain; a **typed** member is an arbitrary value (e.g. an
    address, a tranche id) carried on an element. Both must be rendered: dropping
    typed members would make two facts that differ *only* by a typed member share
    one ``fact_id``, silently losing the second on the idempotent write.
    """
    if value.isExplicit:  # type: ignore[attr-defined]
        return str(value.memberQname)  # type: ignore[attr-defined]
    typed = value.typedMember  # type: ignore[attr-defined]
    return str(typed.stringValue) if typed is not None else ""


def _dimensions(context: object) -> str | None:
    """Render a context's dimensions as a stable, total ``axis=member`` string.

    Both explicit and typed dimensions are included (see :func:`_member`), sorted
    by axis so the result is deterministic and safe to hash into a ``fact_id``.
    """
    dims = context.qnameDims  # type: ignore[attr-defined]
    if not dims:
        return None
    parts = [
        f"{axis}={_member(value)}"
        for axis, value in sorted(dims.items(), key=lambda kv: str(kv[0]))
    ]
    return ";".join(parts) or None


def _fact_id(filing_id: str, concept: str, period: str, dimension: str | None, unit: str) -> str:
    """Derive a deterministic fact id, so re-ingesting a filing is idempotent."""
    key = f"{filing_id}|{concept}|{period}|{dimension or ''}|{unit}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _facts_from_model(model: object, filing_id: str) -> list[FinancialFact]:
    """Map a loaded model's numeric facts to deduplicated :class:`FinancialFact`s.

    Non-numeric facts (DEI metadata) are read separately by :func:`_dei_from_model`;
    nil and contextless facts are dropped; facts repeated across statements
    collapse to one (keyed on concept/period/dimension/unit).
    """
    facts: list[FinancialFact] = []
    seen: set[tuple[str, str, str | None, str]] = set()
    for fact in model.facts:  # type: ignore[attr-defined]
        if not fact.isNumeric or fact.context is None or fact.concept is None:
            continue
        if fact.xValue is None:  # nil-valued fact
            continue
        concept = str(fact.qname)
        period = _period(fact.context)
        dimension = _dimensions(fact.context)
        unit = _unit(fact)
        key = (concept, period, dimension, unit)
        if key in seen:
            continue
        seen.add(key)
        facts.append(
            FinancialFact(
                fact_id=_fact_id(filing_id, concept, period, dimension, unit),
                filing_id=filing_id,
                concept=concept,
                value=float(fact.xValue),
                unit=unit,
                period=period,
                dimension=dimension,
                origin=FactOrigin.XBRL,
            )
        )
    return facts


def _dei_from_model(model: object) -> dict[str, str]:
    """Collect a filing's entity-level DEI values as ``{local_name: value}``.

    DEI facts carry the filing's self-describing metadata (company, CIK, ticker,
    document type, fiscal period). Only undimensioned (entity-level) values are
    kept — a DEI value reported under a segment is not the filing's own — and the
    first value seen for a concept wins.
    """
    dei: dict[str, str] = {}
    for fact in model.facts:  # type: ignore[attr-defined]
        if fact.concept is None or fact.context is None or fact.context.qnameDims:
            continue
        qname = fact.qname
        if _DEI_NS_MARKER not in (qname.namespaceURI or ""):
            continue
        value = fact.value
        if value:
            dei.setdefault(qname.localName, str(value).strip())
    return dei


def _to_int(value: str | None) -> int | None:
    """Parse an integer DEI value (e.g. fiscal year ``"2024"``), or ``None``."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _build_metadata(
    dei: dict[str, str], filing_id: str
) -> tuple[CollectionMetadata, FilingMetadata]:
    """Build the collection and filing rows from a filing's DEI values.

    The collection is keyed on CIK (one company = one collection); a filing
    without a CIK falls back to a per-filing collection so its facts still bind.
    """
    cik = dei.get("EntityCentralIndexKey")
    collection_id = f"us-cik-{cik}" if cik else f"us-{filing_id}"
    collection = CollectionMetadata(
        collection_id=collection_id,
        company=dei.get("EntityRegistrantName"),
        ticker=dei.get("TradingSymbol"),
        cik=cik,
        market=Market.US,
    )
    filing = FilingMetadata(
        filing_id=filing_id,
        collection_id=collection_id,
        filing_type=dei.get("DocumentType"),
        fiscal_year=_to_int(dei.get("DocumentFiscalYearFocus")),
        fiscal_period=dei.get("DocumentFiscalPeriodFocus"),
    )
    return collection, filing


def extract_filing(instance_path: Path, filing_id: str) -> XbrlExtraction:
    """Extract a filing's metadata and numeric facts from a local XBRL instance.

    A single Arelle load yields both: the DEI-derived collection/filing rows and
    the deduplicated numeric facts. The instance's schema and linkbases must sit
    beside it on disk; Arelle resolves them from the document's references.

    Args:
      instance_path: Path to the instance document (inline-XBRL ``.htm`` or a
        standalone instance ``.xml``).
      filing_id: Identifier the filing and its facts are attributed to.

    Returns:
      The filing's collection, filing row, and facts.

    Raises:
      IngestionError: Arelle could not load the instance.
    """
    cntlr = Cntlr.Cntlr(logFileName="logToBuffer")
    try:
        model = cntlr.modelManager.load(str(instance_path))
        if model is None:
            raise IngestionError(f"arelle could not load XBRL instance: {instance_path}")
        facts = _facts_from_model(model, filing_id)
        dei = _dei_from_model(model)
    finally:
        cntlr.modelManager.close()
        cntlr.close()
    collection, filing = _build_metadata(dei, filing_id)
    logger.info("extracted %d facts + %d DEI fields from %s", len(facts), len(dei), filing_id)
    return XbrlExtraction(collection=collection, filing=filing, facts=facts)


def extract_facts(instance_path: Path, filing_id: str) -> list[FinancialFact]:
    """Extract just the numeric facts of a local XBRL instance (see :func:`extract_filing`)."""
    return extract_filing(instance_path, filing_id).facts


def _filing_id(key: str) -> str:
    """Derive a filing identifier from an object key (its accession folder)."""
    return Path(key).parent.name.removesuffix("-xbrl") or Path(key).stem


class ArelleXbrlExtractor:
    """A :class:`~raglearn.core.interfaces.StructuredExtractor` backed by Arelle.

    Assembles the filing's bundle from the object store, then extracts its
    metadata and numeric facts in one load. The instance document is what reaches
    here; its schema and linkbases are pulled in alongside it.
    """

    def __init__(self, assembler: BundleAssembler) -> None:
        """Bind the extractor to a bundle assembler.

        Args:
          assembler: Materializes the filing's bundle to local disk.
        """
        self._assembler = assembler

    def extract(self, document: RawDocument) -> XbrlExtraction | None:
        """Extract the XBRL filing the document belongs to.

        Args:
          document: The XBRL instance document (inline ``.htm`` or standalone
            ``.xml``) that triggered extraction.

        Returns:
          The filing's extraction, or ``None`` if its bundle is not yet complete
          (a later sibling event will retry).
        """
        filing_id = _filing_id(document.doc_id)
        with self._assembler.materialize(document) as instance_path:
            if instance_path is None:
                return None  # bundle incomplete; deferred until its siblings arrive
            return extract_filing(instance_path, filing_id)
