"""Tests for the DuckDB structured store: atomic writes, FK integrity, idempotency.

Hermetic: an in-memory (or tmp-file) DuckDB, no external services. State is
verified by querying the connection directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from raglearn.core.types import (
    CollectionMetadata,
    FactOrigin,
    FilingMetadata,
    FinancialFact,
    Market,
    XbrlExtraction,
)
from raglearn.stores.duckdb_structured import DuckdbStructuredStore


def _fact(fact_id: str = "f1", value: float = 100.0, **overrides: Any) -> FinancialFact:
    fields: dict[str, Any] = {
        "fact_id": fact_id,
        "filing_id": "acc-1",
        "concept": "mock:Revenue",
        "value": value,
        "unit": "USD",
        "period": "2024-01-01/2024-12-31",
        "dimension": None,
        "origin": FactOrigin.XBRL,
    }
    fields.update(overrides)
    return FinancialFact(**fields)


def _extraction(
    facts: list[FinancialFact],
    *,
    collection_id: str = "us-cik-1234567",
    filing_id: str = "acc-1",
    **filing_overrides: Any,
) -> XbrlExtraction:
    collection = CollectionMetadata(
        collection_id=collection_id,
        company="Mock Corp",
        ticker="MOCK",
        cik="1234567",
        market=Market.US,
    )
    filing = FilingMetadata(
        filing_id=filing_id,
        collection_id=collection_id,
        filing_type="10-K",
        fiscal_year=2024,
        fiscal_period="FY",
        **filing_overrides,
    )
    return XbrlExtraction(collection=collection, filing=filing, facts=facts)


def _count(store: DuckdbStructuredStore, table: str) -> int:
    return int(store._conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


@pytest.fixture
def store() -> Iterator[DuckdbStructuredStore]:
    s = DuckdbStructuredStore(":memory:")
    yield s
    s.close()


def test_write_persists_facts_and_their_parents(store: DuckdbStructuredStore) -> None:
    assert store.write(_extraction([_fact("a"), _fact("b")])) == 2
    assert _count(store, "financial_facts") == 2
    assert _count(store, "filings") == 1
    assert _count(store, "collections") == 1


def test_write_with_no_facts_still_writes_parents(store: DuckdbStructuredStore) -> None:
    assert store.write(_extraction([])) == 0
    assert _count(store, "financial_facts") == 0
    assert _count(store, "filings") == 1
    assert _count(store, "collections") == 1


def test_reingesting_a_filing_is_idempotent(store: DuckdbStructuredStore) -> None:
    store.write(_extraction([_fact("a")]))
    store.write(_extraction([_fact("a")]))  # same keys again
    assert _count(store, "financial_facts") == 1
    assert _count(store, "filings") == 1
    assert _count(store, "collections") == 1


def test_conflict_leaves_the_first_fact_value_untouched(store: DuckdbStructuredStore) -> None:
    store.write(_extraction([_fact("a", value=100.0)]))
    store.write(_extraction([_fact("a", value=999.0)]))  # DO NOTHING — original wins
    value = store._conn.execute("SELECT value FROM financial_facts WHERE fact_id = 'a'").fetchone()
    assert value[0] == 100.0


def test_large_integer_value_round_trips_exactly(store: DuckdbStructuredStore) -> None:
    store.write(_extraction([_fact("a", value=123456789000.0)]))
    value = store._conn.execute("SELECT value FROM financial_facts WHERE fact_id = 'a'").fetchone()
    assert value[0] == 123456789000.0


def test_filing_metadata_round_trips(store: DuckdbStructuredStore) -> None:
    store.write(_extraction([_fact("a")]))
    row = store._conn.execute(
        "SELECT collection_id, filing_type, fiscal_year, fiscal_period, version "
        "FROM filings WHERE filing_id = 'acc-1'"
    ).fetchone()
    assert row == ("us-cik-1234567", "10-K", 2024, "FY", 1)


def test_one_collection_shared_across_filings(store: DuckdbStructuredStore) -> None:
    store.write(_extraction([_fact("a")], filing_id="acc-1"))
    store.write(_extraction([_fact("b", filing_id="acc-2")], filing_id="acc-2"))
    assert _count(store, "filings") == 2
    assert _count(store, "collections") == 1  # same company → one collection


def test_foreign_key_rejects_an_orphan_fact(store: DuckdbStructuredStore) -> None:
    # Bypass write() to prove the schema's FK is real: a fact with no filing row.
    with pytest.raises(Exception):  # noqa: B017 (duckdb raises a constraint error type)
        store._conn.execute(
            "INSERT INTO financial_facts VALUES "
            "('x', 'no-such-filing', 'c', 1.0, 'USD', 'p', NULL, 'xbrl')"
        )


def test_facts_persist_across_reopen(tmp_path: Path) -> None:
    path = str(tmp_path / "structured.duckdb")
    first = DuckdbStructuredStore(path)
    first.write(_extraction([_fact("a")]))
    first.close()

    second = DuckdbStructuredStore(path)
    try:
        assert _count(second, "financial_facts") == 1
        assert _count(second, "filings") == 1
    finally:
        second.close()
