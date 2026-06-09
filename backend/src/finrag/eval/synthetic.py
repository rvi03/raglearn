"""A fully synthetic narrative corpus and its golden set for retrieval eval.

Everything here is invented — no real company, filing, or figure — so it can be
generated locally and freely without any data-licensing concern. It is produced
*in code* (not stored as a committed dataset): calling :func:`synthetic_corpus`
rebuilds the same chunks and golden cases deterministically every time.

The corpus is deliberately built to make retrieval metrics *discriminate*:

  * two companies' "annual reports", several chunks each,
  * **distractors** — both companies have a revenue chunk and a risk chunk, so a
    retriever must pick the right company's chunk, not just the right topic,
  * **figures** planted in specific chunks, so numeric-match and citation
    correctness have something concrete to check.

Chunk ids are fixed and stable (not content-hashed) so the golden set can
reference them directly. This evaluates the retrieve → rerank → answer path on
controlled inputs; it does not exercise the real chunker (that needs real
documents and is out of scope for a synthetic set).
"""

from __future__ import annotations

from finrag.core.types import (
    Chunk,
    ChunkType,
    CollectionMetadata,
    DocumentMetadata,
    FactOrigin,
    FilingMetadata,
    FinancialFact,
    Market,
    XbrlExtraction,
)
from finrag.eval.golden import GoldenCase


def _chunk(
    chunk_id: str, text: str, *, company: str, doc_id: str, section: str, page: int
) -> Chunk:
    """Build one synthetic chunk with full provenance metadata."""
    metadata = DocumentMetadata(
        # Same collection id the company's facts use, so entity-scoping a query
        # narrows the narrative search to this company's chunks.
        collection_id=_COLLECTION_BY_COMPANY[company],
        company_name=company,
        market=Market.US,
        filing_type="Annual Report",
        fiscal_year=2024,
        fiscal_period="FY2024",
        section=section,
        source_doc_id=doc_id,
        page=page,
    )
    return Chunk(chunk_id=chunk_id, text=text, chunk_type=ChunkType.TEXT, metadata=metadata)


# --- the two synthetic "filings" ---------------------------------------------
_NORTHWIND = "Northwind Trading Co."
_NW_DOC = "synthetic/northwind-ar-fy2024"
_NW_COLLECTION = "us-cik-901"
_NW_FILING = "nw-fy2024-10k"
_ACME = "Acme Robotics Inc."
_AC_DOC = "synthetic/acme-ar-fy2024"
_AC_COLLECTION = "us-cik-902"
_AC_FILING = "ac-fy2024-10k"
# The annual (FY2024) reporting period shared by the synthetic facts.
_FY2024 = "2023-10-01/2024-09-30"
# Chunks and facts of a company share one collection id (as real ingestion does).
_COLLECTION_BY_COMPANY = {_NORTHWIND: _NW_COLLECTION, _ACME: _AC_COLLECTION}


def synthetic_facts() -> list[XbrlExtraction]:
    """Return synthetic structured filings for the exact (DuckDB) path.

    Mirrors the two corpus companies with a handful of exact XBRL facts (full-dollar
    values, as filings tag them). Ingested into the structured store so the exact
    path can answer figure questions; entirely invented, like the chunk corpus.
    """
    return [
        XbrlExtraction(
            collection=CollectionMetadata(
                collection_id=_NW_COLLECTION,
                company=_NORTHWIND,
                ticker="NWT",
                cik="901",
                market=Market.US,
            ),
            filing=FilingMetadata(
                filing_id=_NW_FILING,
                collection_id=_NW_COLLECTION,
                filing_type="10-K",
                fiscal_year=2024,
            ),
            facts=[
                FinancialFact(
                    fact_id="nw-fact-rev-2024",
                    filing_id=_NW_FILING,
                    concept="us-gaap:Revenues",
                    value=4_250_000_000.0,
                    unit="USD",
                    period=_FY2024,
                    origin=FactOrigin.XBRL,
                ),
                FinancialFact(
                    fact_id="nw-fact-ni-2024",
                    filing_id=_NW_FILING,
                    concept="us-gaap:NetIncomeLoss",
                    value=612_000_000.0,
                    unit="USD",
                    period=_FY2024,
                    origin=FactOrigin.XBRL,
                ),
            ],
        ),
        XbrlExtraction(
            collection=CollectionMetadata(
                collection_id=_AC_COLLECTION,
                company=_ACME,
                ticker="ACME",
                cik="902",
                market=Market.US,
            ),
            filing=FilingMetadata(
                filing_id=_AC_FILING,
                collection_id=_AC_COLLECTION,
                filing_type="10-K",
                fiscal_year=2024,
            ),
            facts=[
                FinancialFact(
                    fact_id="ac-fact-rev-2024",
                    filing_id=_AC_FILING,
                    concept="us-gaap:Revenues",
                    value=880_000_000.0,
                    unit="USD",
                    period=_FY2024,
                    origin=FactOrigin.XBRL,
                ),
            ],
        ),
    ]


def synthetic_corpus() -> tuple[list[Chunk], list[GoldenCase]]:
    """Return the synthetic chunk corpus and the golden cases that score against it.

    Returns:
      A ``(chunks, cases)`` pair. The cases reference chunks by their stable
      ``chunk_id`` and carry the figures a correct answer must state.
    """
    chunks = [
        _chunk(
            "nw-revenue",
            "Northwind Trading reported total revenue of $4,250 million in fiscal year 2024, "
            "up from $3,900 million in fiscal 2023.",
            company=_NORTHWIND,
            doc_id=_NW_DOC,
            section="Financial Review",
            page=12,
        ),
        _chunk(
            # Near-duplicate distractor: same company + metric, the *other* year.
            # A "fiscal 2024 revenue" query must not retrieve/cite this one.
            "nw-revenue-fy23",
            "Northwind Trading reported total revenue of $3,900 million in fiscal year 2023, "
            "the year before the fiscal 2024 result.",
            company=_NORTHWIND,
            doc_id=_NW_DOC,
            section="Financial Review",
            page=11,
        ),
        _chunk(
            "nw-segments",
            "Within fiscal 2024 revenue, the Logistics segment contributed $2,100 million and "
            "the Retail segment contributed $2,150 million.",
            company=_NORTHWIND,
            doc_id=_NW_DOC,
            section="Segment Results",
            page=14,
        ),
        _chunk(
            "nw-dividend",
            "The board of Northwind Trading declared a dividend of $1.20 per share "
            "for fiscal 2024.",
            company=_NORTHWIND,
            doc_id=_NW_DOC,
            section="Capital Returns",
            page=20,
        ),
        _chunk(
            "nw-risk",
            "Principal risks for Northwind include fuel-price volatility and customer "
            "concentration within the Logistics segment.",
            company=_NORTHWIND,
            doc_id=_NW_DOC,
            section="Risk Factors",
            page=28,
        ),
        _chunk(
            "ac-revenue",
            "Acme Robotics generated net revenue of $880 million in fiscal 2024, compared with "
            "$1,020 million in fiscal 2023, a decline driven by softer industrial demand.",
            company=_ACME,
            doc_id=_AC_DOC,
            section="Financial Review",
            page=9,
        ),
        _chunk(
            "ac-rnd",
            "Acme Robotics research and development expense rose to $145 million in fiscal 2024, "
            "reflecting investment in autonomous navigation.",
            company=_ACME,
            doc_id=_AC_DOC,
            section="Operating Expenses",
            page=11,
        ),
        _chunk(
            "ac-guidance",
            "Management expects Acme Robotics fiscal 2025 revenue between $900 million and "
            "$950 million.",
            company=_ACME,
            doc_id=_AC_DOC,
            section="Outlook",
            page=15,
        ),
        _chunk(
            "ac-risk",
            "Key risks for Acme Robotics include supply-chain disruption for semiconductor "
            "components and intense competition from larger automation vendors.",
            company=_ACME,
            doc_id=_AC_DOC,
            section="Risk Factors",
            page=22,
        ),
    ]

    cases = [
        # --- EXACT path: metric queries route to the DuckDB facts (fact ids) ----
        GoldenCase(
            id="x-nw-revenue",
            query="What was Northwind Trading's total revenue in fiscal 2024?",
            query_type="exact",
            expected_chunk_ids=["nw-fact-rev-2024"],
            expected_values=["4,250,000,000"],
        ),
        GoldenCase(
            id="x-nw-netincome",
            query="What was Northwind Trading's net income in fiscal 2024?",
            query_type="exact",
            expected_chunk_ids=["nw-fact-ni-2024"],
            expected_values=["612,000,000"],
        ),
        GoldenCase(
            id="x-ac-revenue",
            query="What was Acme Robotics' total revenue in fiscal 2024?",
            query_type="exact",
            expected_chunk_ids=["ac-fact-rev-2024"],
            expected_values=["880,000,000"],
        ),
        # --- NARRATIVE path: worded to avoid metric keywords so they route narrative
        GoldenCase(
            id="nw-segments",
            query="How did Northwind's Logistics and Retail segments perform in fiscal 2024?",
            expected_chunk_ids=["nw-segments"],
            expected_values=["2,100", "2,150"],
        ),
        GoldenCase(
            id="nw-dividend",
            query="What dividend per share did Northwind Trading declare for fiscal 2024?",
            expected_chunk_ids=["nw-dividend"],
            expected_values=["1.20"],
        ),
        GoldenCase(
            id="nw-risk",
            query="What are the principal risks facing Northwind Trading?",
            expected_chunk_ids=["nw-risk"],
        ),
        GoldenCase(
            id="ac-rnd",
            query="How much did Acme Robotics spend on research and development in fiscal 2024?",
            expected_chunk_ids=["ac-rnd"],
            expected_values=["145"],
        ),
        GoldenCase(
            id="ac-guidance",
            query="What is Acme Robotics' outlook for fiscal 2025?",
            expected_chunk_ids=["ac-guidance"],
            expected_values=["900", "950"],
        ),
        GoldenCase(
            # Negative case: the corpus never names a CEO, so the only correct
            # behaviour is to abstain. Answering this is a hallucination.
            id="nw-ceo",
            query="Who is the chief executive officer of Northwind Trading?",
            answerable=False,
        ),
    ]
    return chunks, cases
