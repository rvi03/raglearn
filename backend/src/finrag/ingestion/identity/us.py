"""US identity from XBRL DEI.

A US filing's identity is already machine-readable: the DEI facts the facts arm
extracts via Arelle carry the registrant, CIK, form type, and fiscal period. So
this adapter does not re-read the document — it maps the
:class:`~finrag.core.types.XbrlExtraction` the facts arm produced into a
:class:`~finrag.core.types.DocumentIdentity`. Identity is therefore exact and
high-confidence; no classification is involved.
"""

from __future__ import annotations

from pathlib import Path

from finrag.core.errors import IngestionError
from finrag.core.registry import registry
from finrag.core.types import (
    DocumentIdentity,
    Market,
    NumericAuthority,
    ParsedStructure,
    RawDocument,
    XbrlExtraction,
)
from finrag.ingestion.identity import rules


def identity_from_extraction(extraction: XbrlExtraction, document: RawDocument) -> DocumentIdentity:
    """Map a filing's extracted DEI (collection + filing) into its identity."""
    collection = extraction.collection
    filing = extraction.filing
    form = filing.filing_type or rules.UNKNOWN
    period = filing.fiscal_period or (str(filing.fiscal_year) if filing.fiscal_year else None)
    return DocumentIdentity(
        market=Market.US,
        collection_id=collection.collection_id,
        logical_key=rules.logical_key(collection.collection_id, rules.form_family(form), period),
        content_hash=rules.content_hash(document.data or b""),
        doc_type=form,
        numeric_authority=(
            rules.numeric_authority_for_form(form)
            if filing.filing_type
            else NumericAuthority.INDICATIVE
        ),
        company=collection.company,
        cik=collection.cik,
        ticker=collection.ticker,
        fiscal_year=filing.fiscal_year,
        fiscal_period=filing.fiscal_period,
        recency=filing.filed_date,
        confidence=1.0,
        needs_review=filing.filing_type is None,
    )


def _minimal_identity(document: RawDocument) -> DocumentIdentity:
    """Best-effort identity for a US page document with no DEI (an exhibit/image).

    Exhibits and images carry no DEI, so identity is grouped by their accession
    folder and flagged for review; they belong to a filing but are not the
    authoritative source of its figures.
    """
    accession = (
        Path(document.doc_id).parent.name.removesuffix("-xbrl") or Path(document.doc_id).stem
    )
    collection_id = f"us-{accession}"
    return DocumentIdentity(
        market=Market.US,
        collection_id=collection_id,
        logical_key=rules.logical_key(collection_id, "exhibit", None),
        content_hash=rules.content_hash(document.data or b""),
        doc_type="exhibit",
        numeric_authority=NumericAuthority.INDICATIVE,
        confidence=0.3,
        needs_review=True,
    )


@registry.register("identity_extractor", "us")
class UsIdentityExtractor:
    """An :class:`~finrag.core.interfaces.IdentityExtractor` for US (EDGAR) filings."""

    def identify(
        self,
        document: RawDocument,
        *,
        structure: ParsedStructure | None = None,
        extraction: XbrlExtraction | None = None,
    ) -> DocumentIdentity:
        """Return the filing's identity from its XBRL extraction.

        Args:
          document: The triggering document (its bytes give the content hash).
          structure: Used only for the no-DEI page case (an exhibit/image),
            where it signals a best-effort, review-flagged identity.
          extraction: The filing's XBRL extraction (DEI + facts) from the facts
            arm. When present, identity is read from the DEI.

        Raises:
          IngestionError: Neither an extraction nor a parsed structure was given.
        """
        if extraction is not None:
            return identity_from_extraction(extraction, document)
        if structure is not None:
            return _minimal_identity(document)
        raise IngestionError(
            "us identity_extractor requires an XBRL extraction or a parsed structure"
        )
