"""LLM text-to-SQL structured QA: the fallback for off-registry fact questions.

The metric registry only answers metrics it has been taught. This adapter widens
the exact path: it asks the LLM to write a ``SELECT`` over the facts schema for a
question the registry could not map, runs it through the SQL guard, executes it
read-only, and returns the matching facts. It is a *fallback*, never the
authority — the registry is tried first (see :class:`FallbackStructuredQA`).

Safety is delegated to :func:`finrag.retrieval.sql_guard.validate_select`: the
model only ever influences the filtering, and any SQL that is not a safe,
allow-listed ``SELECT`` makes this adapter **abstain** (return ``[]``) so the
caller falls back to narrative — a confidently wrong figure is worse than none.
The generation runs under ``suppressed()`` so the SQL never leaks into the user's
token stream.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from finrag.core.interfaces.generation import LLMBackend
from finrag.core.logging import get_logger
from finrag.core.registry import registry
from finrag.core.types import FinancialFact, Query
from finrag.generation.token_stream import suppressed
from finrag.retrieval.sql_guard import SqlGuardError, validate_select

logger = get_logger(__name__)

# The schema the model writes against — the three allow-listed tables and the
# columns the guard permits, plus the facts needed to filter correctly.
_SCHEMA = """\
financial_facts(fact_id, filing_id, concept, value, unit, period, dimension, origin)
filings(filing_id, collection_id, filing_type, fiscal_year, fiscal_period)
collections(collection_id, company, ticker, cik, market)
joins: financial_facts.filing_id -> filings.filing_id -> collections.collection_id"""


@runtime_checkable
class _FactQueryRunner(Protocol):
    """The slice of the structured store this adapter executes SQL through."""

    def select_facts(self, sql: str) -> list[FinancialFact]: ...


def _build_prompt(question: str, collection_id: str) -> str:
    """Build the text-to-SQL prompt, scoped to the already-resolved company."""
    return (
        "You translate a question about a company's financial filings into ONE "
        "read-only SQL SELECT over this schema:\n\n"
        f"{_SCHEMA}\n\n"
        "Rules:\n"
        "- One SELECT only; read from financial_facts (join filings/collections to filter).\n"
        f"- Scope to this company: filings.collection_id = '{collection_id}'.\n"
        "- `concept` holds XBRL tags (e.g. 'us-gaap:NetIncomeLoss').\n"
        "- `period` is 'YYYY-MM-DD/YYYY-MM-DD' for a duration or a single date for an instant.\n"
        "- Company-level (undimensioned) facts have dimension IS NULL.\n"
        "- No INSERT/UPDATE/DELETE/DDL, no other tables.\n\n"
        f"Question: {question}\n\n"
        "Return only the SQL, nothing else."
    )


def _extract_sql(text: str) -> str:
    """Pull the SELECT out of the model's reply (strip prose and ``` fences)."""
    cleaned = text.replace("```sql", "```").replace("```", "\n").strip()
    lowered = cleaned.lower()
    start = lowered.find("select")
    if start == -1:
        return ""
    return cleaned[start:].strip().rstrip(";").strip()


@registry.register("structured_qa", "text_to_sql")
class TextToSqlQA:
    """A :class:`~finrag.core.interfaces.StructuredQA` that generates guarded SQL."""

    def __init__(self, llm: LLMBackend, store: _FactQueryRunner) -> None:
        """Bind the adapter to the LLM that writes SQL and the store that runs it."""
        self._llm = llm
        self._store = store

    def answer(self, query: Query) -> list[FinancialFact]:
        """Generate, guard, and run a SELECT for the query; abstain (``[]``) on any failure.

        Requires a resolved ``collection_id`` in the query filters (the caller sets
        it before routing here). Returns ``[]`` — never raises — so the caller falls
        back to narrative when the SQL is unsafe, unparseable, or yields nothing.
        """
        collection_id = query.filters.get("collection_id")
        if not collection_id:
            return []
        try:
            with suppressed():  # keep the generated SQL out of the user's token stream
                raw = self._llm.generate(_build_prompt(query.text, collection_id)).text
            sql = _extract_sql(raw)
            if not sql:
                return []
            return self._store.select_facts(validate_select(sql))
        except SqlGuardError as exc:
            logger.info("text_to_sql rejected unsafe SQL: %s", exc)
            return []
        except Exception:
            logger.exception("text_to_sql failed; abstaining")
            return []
