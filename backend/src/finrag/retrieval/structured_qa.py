"""Metric-registry structured QA: exact figures from the DuckDB facts.

The exact path's core. A natural-language question naming a financial metric and
(optionally) a year is resolved to:

  * a **metric** -> its candidate XBRL concept(s) (a curated registry, not
    LLM-generated SQL — deterministic and safe),
  * a **company** -> its collection (entity resolution in the store),
  * a **year** -> the fact whose period ends in that fiscal year,

and the single matching undimensioned fact is returned. No LLM is involved, so
the figure is exactly as filed. The registry is intentionally small and explicit;
unmapped metrics simply return nothing (the caller falls back to narrative).

The store is reached through a narrow :class:`_FactReader` protocol so this stage
does not depend on the concrete DuckDB store.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Protocol, runtime_checkable

from finrag.core.registry import registry
from finrag.core.types import FinancialFact, Query

# Canonical metric -> candidate XBRL concepts, in preference order (a metric is
# tagged differently across filers/years, so we try several).
_METRIC_CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:Revenues",
        "us-gaap:SalesRevenueNet",
    ],
    "net income": ["us-gaap:NetIncomeLoss"],
    "gross profit": ["us-gaap:GrossProfit"],
    "operating income": ["us-gaap:OperatingIncomeLoss"],
    "total assets": ["us-gaap:Assets"],
    "diluted eps": ["us-gaap:EarningsPerShareDiluted"],
    "basic eps": ["us-gaap:EarningsPerShareBasic"],
}

# Query phrase -> canonical metric. Longest phrases are matched first so
# "net income" beats "income" and "total assets" beats "assets".
_SYNONYMS: dict[str, str] = {
    "total revenue": "revenue",
    "net sales": "revenue",
    "revenue": "revenue",
    "sales": "revenue",
    "net income": "net income",
    "net earnings": "net income",
    "net profit": "net income",
    "gross profit": "gross profit",
    "operating income": "operating income",
    "operating profit": "operating income",
    "total assets": "total assets",
    "diluted eps": "diluted eps",
    "diluted earnings per share": "diluted eps",
    "basic eps": "basic eps",
    "basic earnings per share": "basic eps",
    "earnings per share": "diluted eps",
    "eps": "diluted eps",
}

# A 4-digit year, optionally prefixed by FY / "fiscal".
_YEAR_RE = re.compile(r"\b(?:fy\s?|fiscal\s+)?((?:19|20)\d{2})\b")
# Minimum span (days) for a duration period to count as annual vs quarterly.
_ANNUAL_MIN_DAYS = 300


@runtime_checkable
class _FactReader(Protocol):
    """The slice of the structured store the metric QA reads through."""

    def find_collection(self, text: str) -> tuple[str, str] | None: ...

    def query_facts(self, collection_id: str, concepts: list[str]) -> list[FinancialFact]: ...


def resolve_metric(text: str) -> tuple[str, list[str]] | None:
    """Resolve a metric named in the query to ``(canonical_name, concepts)``.

    Matches the longest synonym phrase present, so specific phrases win over their
    substrings. Returns ``None`` when no known metric is mentioned.
    """
    lowered = text.lower()
    for phrase in sorted(_SYNONYMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            canonical = _SYNONYMS[phrase]
            return canonical, _METRIC_CONCEPTS[canonical]
    return None


def parse_year(text: str) -> int | None:
    """Return the fiscal year named in the query (``FY2023``/``2023``), or ``None``."""
    match = _YEAR_RE.search(text.lower())
    return int(match.group(1)) if match else None


def _period_end(period: str) -> str:
    """Return a period's end date string (after ``/`` for a duration; the instant)."""
    return period.split("/")[-1]


def _end_year(period: str) -> int | None:
    """Return the year of a period's end date, or ``None`` if unparseable."""
    end = _period_end(period)
    try:
        return date.fromisoformat(end).year
    except ValueError:
        return None


def _is_annual(period: str) -> bool:
    """Return whether a duration period spans roughly a year (vs a quarter)."""
    if "/" not in period:
        return False  # an instant (balance-sheet) period, not a duration
    start, end = period.split("/", 1)
    try:
        span = (date.fromisoformat(end) - date.fromisoformat(start)).days
    except ValueError:
        return False
    return span >= _ANNUAL_MIN_DAYS


def _select(facts: list[FinancialFact], year: int | None) -> FinancialFact | None:
    """Pick the single fact that best answers the query.

    With a year, restrict to facts ending in that year and abstain if none match
    (better no answer than the wrong year). Then prefer annual durations over
    quarterly, and the most recent period.
    """
    candidates = facts
    if year is not None:
        candidates = [f for f in facts if _end_year(f.period) == year]
        if not candidates:
            return None
    if not candidates:
        return None
    candidates.sort(key=lambda f: (_is_annual(f.period), _period_end(f.period)), reverse=True)
    return candidates[0]


@registry.register("structured_qa", "metric_registry")
class MetricRegistryQA:
    """A :class:`~finrag.core.interfaces.StructuredQA` over the DuckDB facts."""

    def __init__(self, store: _FactReader) -> None:
        """Bind the QA to the structured store it reads facts from."""
        self._store = store

    def answer(self, query: Query) -> list[FinancialFact]:
        """Return the exact fact(s) answering the query, or ``[]`` if none apply.

        Returns ``[]`` (not an error) when the query names no known metric, no
        company resolves, or no fact matches the requested year — the caller then
        falls back to the narrative path.
        """
        metric = resolve_metric(query.text)
        if metric is None:
            return []
        _name, concepts = metric

        collection_id = query.filters.get("collection_id")
        if collection_id is None:
            found = self._store.find_collection(query.text)
            if found is None:
                return []
            collection_id = found[0]

        year = parse_year(query.text)
        for concept in concepts:  # preference order: first tagging that yields a fact wins
            selected = _select(self._store.query_facts(collection_id, [concept]), year)
            if selected is not None:
                return [selected]
        return []
