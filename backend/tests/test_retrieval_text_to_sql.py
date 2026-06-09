"""Tests for LLM text-to-SQL QA and the metric→SQL fallback chain.

Hermetic: a fake LLM supplies the SQL, an in-memory DuckDB store executes it. No
Ollama, no external services.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from finrag.core.types import (
    CollectionMetadata,
    FactOrigin,
    FilingMetadata,
    FinancialFact,
    LLMResponse,
    Market,
    Query,
    XbrlExtraction,
)
from finrag.retrieval.composite_qa import FallbackStructuredQA
from finrag.retrieval.text_to_sql import TextToSqlQA
from finrag.stores.duckdb_structured import DuckdbStructuredStore

_COLLECTION = "us-cik-1234567"


class _FakeLLM:
    """Returns canned SQL text regardless of the prompt."""

    def __init__(self, sql: str) -> None:
        self._sql = sql

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(text=self._sql, model="fake")


@pytest.fixture
def store() -> Iterator[DuckdbStructuredStore]:
    s = DuckdbStructuredStore(":memory:")
    s.write(
        XbrlExtraction(
            collection=CollectionMetadata(
                collection_id=_COLLECTION,
                company="Mock Corp",
                ticker="MOCK",
                cik="1234567",
                market=Market.US,
            ),
            filing=FilingMetadata(
                filing_id="acc-1",
                collection_id=_COLLECTION,
                filing_type="10-K",
                fiscal_year=2024,
                fiscal_period="FY",
            ),
            facts=[
                FinancialFact(
                    fact_id="f1",
                    filing_id="acc-1",
                    concept="us-gaap:NetIncomeLoss",
                    value=4250.0,
                    unit="USD",
                    period="2024-01-01/2024-12-31",
                    dimension=None,
                    origin=FactOrigin.XBRL,
                )
            ],
        )
    )
    yield s
    s.close()


def _query(text: str = "what was net income") -> Query:
    return Query(text=text, filters={"collection_id": _COLLECTION})


def test_generated_sql_returns_the_matching_fact(store: DuckdbStructuredStore) -> None:
    sql = (
        "SELECT f.value FROM financial_facts f "
        "JOIN filings fl ON f.filing_id = fl.filing_id "
        f"WHERE fl.collection_id = '{_COLLECTION}' AND f.concept = 'us-gaap:NetIncomeLoss'"
    )
    facts = TextToSqlQA(_FakeLLM(sql), store).answer(_query())
    assert [f.value for f in facts] == [4250.0]
    assert facts[0].concept == "us-gaap:NetIncomeLoss"  # full fact reconstructed, not just value


def test_unsafe_sql_makes_it_abstain(store: DuckdbStructuredStore) -> None:
    qa = TextToSqlQA(_FakeLLM("DROP TABLE financial_facts"), store)
    assert qa.answer(_query()) == []  # abstains rather than executing


def test_non_sql_reply_abstains(store: DuckdbStructuredStore) -> None:
    assert TextToSqlQA(_FakeLLM("I cannot help with that."), store).answer(_query()) == []


def test_missing_collection_abstains(store: DuckdbStructuredStore) -> None:
    qa = TextToSqlQA(_FakeLLM("SELECT f.value FROM financial_facts f"), store)
    assert qa.answer(Query(text="net income")) == []  # no collection_id resolved


def test_fenced_sql_is_extracted(store: DuckdbStructuredStore) -> None:
    fenced = (
        "```sql\nSELECT f.value FROM financial_facts f "
        "JOIN filings fl ON f.filing_id = fl.filing_id "
        f"WHERE fl.collection_id = '{_COLLECTION}'\n```"
    )
    facts = TextToSqlQA(_FakeLLM(fenced), store).answer(_query())
    assert [f.value for f in facts] == [4250.0]


class _StubQA:
    def __init__(self, facts: list[FinancialFact]) -> None:
        self._facts = facts

    def answer(self, query: Query) -> list[FinancialFact]:
        return self._facts


def _fact() -> FinancialFact:
    return FinancialFact(
        fact_id="x",
        filing_id="acc-1",
        concept="us-gaap:Revenues",
        value=1.0,
        unit="USD",
        period="2024-01-01/2024-12-31",
        dimension=None,
        origin=FactOrigin.XBRL,
    )


def test_fallback_prefers_the_first_stage_that_answers() -> None:
    registry_hit = _StubQA([_fact()])
    never = _StubQA([])
    # The registry answers, so the SQL stage is never consulted.
    assert FallbackStructuredQA([registry_hit, never]).answer(_query()) == [_fact()]


def test_fallback_uses_later_stage_when_earlier_is_empty() -> None:
    chain = FallbackStructuredQA([_StubQA([]), _StubQA([_fact()])])
    assert chain.answer(_query()) == [_fact()]
