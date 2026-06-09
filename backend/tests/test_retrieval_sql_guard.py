"""Tests for the LLM-SQL safety guard: only safe, allow-listed SELECTs survive."""

from __future__ import annotations

import pytest

from finrag.retrieval.sql_guard import SqlGuardError, validate_select

_GOOD = (
    "SELECT f.value FROM financial_facts f "
    "JOIN filings fl ON f.filing_id = fl.filing_id "
    "WHERE fl.collection_id = 'c1' AND f.concept = 'us-gaap:NetIncomeLoss'"
)


def test_good_select_is_accepted_and_projection_is_canonicalised() -> None:
    out = validate_select(_GOOD)
    # The model never controls the projection — it is overwritten with the fact columns.
    assert out.startswith(
        "SELECT f.fact_id, f.filing_id, f.concept, f.value, f.unit, f.period, f.dimension, f.origin"
    )
    assert "LIMIT 50" in out  # a ceiling is enforced even when the model omits one


def test_select_star_is_rewritten_to_fact_columns() -> None:
    out = validate_select("SELECT * FROM financial_facts f WHERE f.value > 0")
    assert "f.fact_id" in out and "SELECT *" not in out


def test_excessive_limit_is_clamped() -> None:
    out = validate_select(f"{_GOOD} LIMIT 100000")
    assert "LIMIT 50" in out and "100000" not in out


def test_modest_limit_is_kept() -> None:
    assert "LIMIT 5" in validate_select(f"{_GOOD} LIMIT 5")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT f.value FROM financial_facts f; DROP TABLE filings",  # multi-statement
        "DROP TABLE financial_facts",  # DDL
        "UPDATE financial_facts SET value = 0",  # DML
        "DELETE FROM financial_facts",  # DML
        "SELECT * FROM users",  # disallowed table
        "SELECT value FROM filings",  # no financial_facts
        "SELECT f.password FROM financial_facts f",  # disallowed column
        "SELECT f.value FROM financial_facts f WHERE f.value > (SELECT 1)",  # subquery
        "SELECT f.value FROM financial_facts f UNION SELECT cik FROM collections",  # set-op
        "WITH x AS (SELECT 1) SELECT f.value FROM financial_facts f",  # CTE
        "not sql at all",  # unparseable / not a select
    ],
)
def test_unsafe_sql_is_rejected(sql: str) -> None:
    with pytest.raises(SqlGuardError):
        validate_select(sql)
