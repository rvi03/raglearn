"""Tests for the metric-registry exact path (hermetic: in-memory DuckDB).

Covers metric/year resolution, company resolution, period selection (right year,
annual over quarterly), concept-preference, and the abstain cases — over a real
in-memory structured store seeded with synthetic facts.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from finrag.core.interfaces.retrieval import StructuredQA
from finrag.core.registry import registry
from finrag.core.types import (
    CollectionMetadata,
    FactOrigin,
    FilingMetadata,
    FinancialFact,
    Market,
    Query,
    XbrlExtraction,
)
from finrag.retrieval.structured_qa import MetricRegistryQA, parse_year, resolve_metric
from finrag.stores.duckdb_structured import DuckdbStructuredStore


def _fact(fact_id: str, concept: str, value: float, period: str, **kw: object) -> FinancialFact:
    return FinancialFact(
        fact_id=fact_id,
        filing_id=kw.get("filing_id", "nw-10k"),  # type: ignore[arg-type]
        concept=concept,
        value=value,
        unit=kw.get("unit", "USD"),  # type: ignore[arg-type]
        period=period,
        dimension=kw.get("dimension"),  # type: ignore[arg-type]
        origin=FactOrigin.XBRL,
    )


@pytest.fixture
def store() -> Iterator[DuckdbStructuredStore]:
    s = DuckdbStructuredStore(":memory:")
    # Northwind: revenue for two years + net income; a dimensioned (segment) row.
    s.write(
        XbrlExtraction(
            collection=CollectionMetadata(
                collection_id="us-cik-100",
                company="Northwind Trading Co.",
                ticker="NWT",
                cik="100",
                market=Market.US,
            ),
            filing=FilingMetadata(
                filing_id="nw-10k", collection_id="us-cik-100", filing_type="10-K", fiscal_year=2024
            ),
            facts=[
                _fact("nw-rev-24", "us-gaap:Revenues", 4250.0, "2023-10-01/2024-09-30"),
                _fact("nw-rev-23", "us-gaap:Revenues", 3900.0, "2022-10-01/2023-09-30"),
                _fact("nw-ni-24", "us-gaap:NetIncomeLoss", 500.0, "2023-10-01/2024-09-30"),
                # Quarterly + dimensioned distractors that must NOT be returned.
                _fact("nw-rev-q4", "us-gaap:Revenues", 1100.0, "2024-07-01/2024-09-30"),
                _fact(
                    "nw-rev-seg",
                    "us-gaap:Revenues",
                    2000.0,
                    "2023-10-01/2024-09-30",
                    dimension="segment=Logistics",
                ),
            ],
        )
    )
    s.write(
        XbrlExtraction(
            collection=CollectionMetadata(
                collection_id="us-cik-200",
                company="Acme Robotics Inc.",
                ticker="ACME",
                cik="200",
                market=Market.US,
            ),
            filing=FilingMetadata(
                filing_id="ac-10k", collection_id="us-cik-200", filing_type="10-K", fiscal_year=2024
            ),
            facts=[
                _fact(
                    "ac-rev-24",
                    "us-gaap:Revenues",
                    880.0,
                    "2023-10-01/2024-09-30",
                    filing_id="ac-10k",
                )
            ],
        )
    )
    yield s
    s.close()


# --- pure resolvers -----------------------------------------------------------


def test_resolve_metric_maps_synonyms() -> None:
    assert resolve_metric("net sales")[0] == "revenue"  # type: ignore[index]
    assert resolve_metric("what was net income")[0] == "net income"  # type: ignore[index]
    assert resolve_metric("the weather today") is None


def test_parse_year_handles_fy_and_plain() -> None:
    assert parse_year("revenue in FY2023") == 2023
    assert parse_year("fiscal 2024 sales") == 2024
    assert parse_year("total revenue") is None


# --- store reads --------------------------------------------------------------


def test_find_collection_by_name_and_ticker(store: DuckdbStructuredStore) -> None:
    assert store.find_collection("What was Northwind's revenue?") == (
        "us-cik-100",
        "Northwind Trading Co.",
    )
    assert store.find_collection("ACME revenue 2024")[0] == "us-cik-200"  # type: ignore[index]
    assert store.find_collection("Globex revenue") is None


def test_query_facts_excludes_dimensioned(store: DuckdbStructuredStore) -> None:
    facts = store.query_facts("us-cik-100", ["us-gaap:Revenues"])
    ids = {f.fact_id for f in facts}
    assert "nw-rev-seg" not in ids  # dimensioned segment row excluded
    assert {"nw-rev-24", "nw-rev-23", "nw-rev-q4"} <= ids


# --- the QA -------------------------------------------------------------------


def test_answers_revenue_for_requested_year(store: DuckdbStructuredStore) -> None:
    qa = MetricRegistryQA(store)
    [fact] = qa.answer(Query(text="What was Northwind's total revenue in fiscal 2024?"))
    assert fact.value == 4250.0
    assert fact.fact_id == "nw-rev-24"  # annual FY2024, not the quarter or FY2023


def test_other_year_selects_other_fact(store: DuckdbStructuredStore) -> None:
    qa = MetricRegistryQA(store)
    [fact] = qa.answer(Query(text="Northwind revenue in 2023"))
    assert fact.value == 3900.0


def test_unknown_year_abstains(store: DuckdbStructuredStore) -> None:
    qa = MetricRegistryQA(store)
    assert qa.answer(Query(text="Northwind revenue in 2099")) == []  # no wrong-year guess


def test_net_income_metric(store: DuckdbStructuredStore) -> None:
    qa = MetricRegistryQA(store)
    [fact] = qa.answer(Query(text="Northwind net income FY2024"))
    assert fact.value == 500.0


def test_no_metric_or_no_company_abstains(store: DuckdbStructuredStore) -> None:
    qa = MetricRegistryQA(store)
    assert qa.answer(Query(text="What are Northwind's risks?")) == []  # no metric
    assert qa.answer(Query(text="Globex revenue 2024")) == []  # company not found


def test_registered_and_conforms_to_protocol(store: DuckdbStructuredStore) -> None:
    adapter = registry.create("structured_qa", "metric_registry", store=store)
    assert isinstance(adapter, MetricRegistryQA)
    assert isinstance(adapter, StructuredQA)
