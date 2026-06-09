"""Tests for the content-based identity extractors (US + India).

Hermetic: the US adapter is fed a synthetic XbrlExtraction (no Arelle) and the
India adapter a synthetic ParsedStructure (no Docling) — the DEI/title-bound
paths are exercised by out-of-band smokes. These cover the pure mapping: kind,
authority tier, logical key, content hash, the confidence gate, and dispatch.
"""

from __future__ import annotations

import pytest

from finrag.core.errors import IngestionError
from finrag.core.interfaces.ingestion import IdentityExtractor
from finrag.core.types import (
    BlockKind,
    CollectionMetadata,
    FilingMetadata,
    Market,
    NumericAuthority,
    ParsedStructure,
    RawDocument,
    StructureBlock,
    StructureSection,
    XbrlExtraction,
)
from finrag.ingestion.identity import (
    metadata_from_identity,
    resolve_identity_extractor,
    rules,
)
from finrag.ingestion.identity.india import IndiaIdentityExtractor
from finrag.ingestion.identity.us import UsIdentityExtractor

_INDIA_KEY = "india/mockpharma/Q4FY2026/results.pdf"
_US_KEY = "us/mockco/0000001234-23-000106-xbrl/mock-20230930.htm"


def _india_doc(data: bytes = b"pdf-bytes") -> RawDocument:
    return RawDocument(
        doc_id=_INDIA_KEY, filename="results.pdf", content_type="application/pdf", data=data
    )


def _structure(title: str) -> ParsedStructure:
    block = StructureBlock(kind=BlockKind.TEXT, text="body text")
    return ParsedStructure(
        source_doc_id=_INDIA_KEY, sections=[StructureSection(title=title, level=0, blocks=[block])]
    )


def _us_doc(data: bytes = b"htm-bytes") -> RawDocument:
    return RawDocument(
        doc_id=_US_KEY, filename="mock-20230930.htm", content_type="text/html", data=data
    )


def _extraction(filing_type: str | None = "10-K") -> XbrlExtraction:
    coll = CollectionMetadata(
        collection_id="us-cik-0000001234",
        company="Mock Corp",
        ticker="MOCK",
        cik="0000001234",
        market=Market.US,
    )
    filing = FilingMetadata(
        filing_id="0000001234-23-000106",
        collection_id="us-cik-0000001234",
        filing_type=filing_type,
        fiscal_year=2023,
        fiscal_period="FY",
    )
    return XbrlExtraction(collection=coll, filing=filing, facts=[])


# --- India: title rules -------------------------------------------------------


def test_india_financial_results_is_authoritative() -> None:
    title = "Mockpharma — Unaudited Financial Results for the Quarter Ended Mar 31, 2026"
    ident = IndiaIdentityExtractor().identify(_india_doc(), structure=_structure(title))

    assert ident.market is Market.IN
    assert ident.doc_type == rules.FINANCIAL_RESULTS
    assert ident.numeric_authority is NumericAuthority.AUTHORITATIVE
    assert ident.company == "mockpharma"  # weak hint from the path, not the title
    assert ident.content_hash and not ident.needs_review


def test_india_presentation_is_indicative_with_period() -> None:
    ident = IndiaIdentityExtractor().identify(
        _india_doc(), structure=_structure("Investor Presentation — Q4 FY26")
    )

    assert ident.doc_type == rules.INVESTOR_PRESENTATION
    assert ident.numeric_authority is NumericAuthority.INDICATIVE
    assert ident.fiscal_period == "Q4 FY26"


def test_india_transcript_classified_from_title() -> None:
    ident = IndiaIdentityExtractor().identify(
        _india_doc(), structure=_structure("Earnings Conference Call Transcript — May 29, 2026")
    )
    assert ident.doc_type == rules.EARNINGS_CALL_TRANSCRIPT


def test_india_categorises_from_pdf_cover_when_structure_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Structure title abstains (graphic slide -> "Disclaimer"), but the PDF digital
    # text layer carries the real title -> categorisation still succeeds, no LLM.
    monkeypatch.setattr(
        "finrag.ingestion.identity.india.pdf_cover_text",
        lambda _data: "Q4FY26 & FY26 Mockpharma Earnings Presentation",
    )
    ident = IndiaIdentityExtractor().identify(_india_doc(), structure=_structure("Disclaimer"))

    assert ident.doc_type == rules.INVESTOR_PRESENTATION
    assert ident.fiscal_period == "Q4FY26"
    assert not ident.needs_review


def test_india_unknown_title_is_flagged() -> None:
    ident = IndiaIdentityExtractor().identify(_india_doc(), structure=_structure("Internal Memo"))

    assert ident.doc_type == rules.UNKNOWN
    assert ident.needs_review is True
    assert ident.confidence <= 0.3


# --- India: LLM fallback seam + confidence gate -------------------------------


def test_india_llm_fallback_used_only_when_rules_miss() -> None:
    extractor = IndiaIdentityExtractor(classifier=lambda _text: (rules.FINANCIAL_RESULTS, 0.8))
    ident = extractor.identify(_india_doc(), structure=_structure("Some Untemplated Heading"))

    assert ident.doc_type == rules.FINANCIAL_RESULTS
    assert ident.confidence == 0.8
    assert ident.needs_review is False


def test_india_low_confidence_classifier_is_gated() -> None:
    extractor = IndiaIdentityExtractor(classifier=lambda _text: (rules.FINANCIAL_RESULTS, 0.4))
    ident = extractor.identify(_india_doc(), structure=_structure("Some Untemplated Heading"))

    assert ident.doc_type == rules.UNKNOWN
    assert ident.needs_review is True


def test_india_requires_structure() -> None:
    with pytest.raises(IngestionError, match="parsed structure"):
        IndiaIdentityExtractor().identify(_india_doc())


# --- US: DEI mapping ----------------------------------------------------------


def test_us_periodic_form_is_authoritative() -> None:
    ident = UsIdentityExtractor().identify(_us_doc(), extraction=_extraction("10-K"))

    assert ident.market is Market.US
    assert ident.doc_type == "10-K"
    assert ident.numeric_authority is NumericAuthority.AUTHORITATIVE
    assert ident.cik == "0000001234"
    assert "us-cik-0000001234" in ident.logical_key
    assert ident.confidence == 1.0


def test_us_amendment_groups_with_base_form() -> None:
    ident = UsIdentityExtractor().identify(_us_doc(), extraction=_extraction("10-K/A"))

    assert ident.doc_type == "10-K/A"  # preserved as the kind
    assert "|10-K|" in ident.logical_key  # but groups under the base form
    assert ident.numeric_authority is NumericAuthority.AUTHORITATIVE


def test_us_current_report_is_indicative() -> None:
    ident = UsIdentityExtractor().identify(_us_doc(), extraction=_extraction("8-K"))
    assert ident.numeric_authority is NumericAuthority.INDICATIVE


def test_us_requires_extraction_or_structure() -> None:
    with pytest.raises(IngestionError, match="XBRL extraction"):
        UsIdentityExtractor().identify(_us_doc())


def test_us_structure_only_yields_minimal_review_identity() -> None:
    # An exhibit/image has no DEI -> best-effort, accession-grouped, flagged.
    ident = UsIdentityExtractor().identify(_us_doc(), structure=_structure("Exhibit 21"))

    assert ident.market is Market.US
    assert ident.doc_type == "exhibit"
    assert ident.needs_review is True
    assert "0000001234-23-000106" in ident.logical_key


# --- dispatch + hashing + conformance -----------------------------------------


def test_dispatch_by_country_prefix() -> None:
    assert isinstance(resolve_identity_extractor(_US_KEY), UsIdentityExtractor)
    assert isinstance(resolve_identity_extractor(_INDIA_KEY), IndiaIdentityExtractor)


def test_dispatch_unknown_country_rejected() -> None:
    with pytest.raises(IngestionError, match="country prefix"):
        resolve_identity_extractor("brazil/petrobras/2026/ar.pdf")


def test_content_hash_is_deterministic_and_distinct() -> None:
    assert rules.content_hash(b"same") == rules.content_hash(b"same")
    assert rules.content_hash(b"a") != rules.content_hash(b"b")


def test_adapters_conform_to_protocol() -> None:
    assert isinstance(UsIdentityExtractor(), IdentityExtractor)
    assert isinstance(IndiaIdentityExtractor(), IdentityExtractor)


# --- identity -> metadata mapper ----------------------------------------------


def test_metadata_carries_version_fields() -> None:
    ident = UsIdentityExtractor().identify(_us_doc(), extraction=_extraction("10-K"))
    md = metadata_from_identity(ident, _US_KEY, section="Item 1A", page=3)

    assert md.collection_id == "us-cik-0000001234"
    assert md.filing_type == "10-K"
    assert md.content_hash == ident.content_hash
    assert md.logical_key == ident.logical_key
    assert md.numeric_authority is NumericAuthority.AUTHORITATIVE
    assert md.section == "Item 1A"
    assert md.page == 3
    assert md.source_doc_id == _US_KEY


def test_metadata_india_company_falls_back_to_hint() -> None:
    ident = IndiaIdentityExtractor().identify(_india_doc(), structure=_structure("Internal Memo"))
    md = metadata_from_identity(ident, _INDIA_KEY)

    assert md.market is Market.IN
    assert md.collection_id == "in-mockpharma"
    assert md.company_name == "mockpharma"  # folder hint, never the literal "unknown" here
