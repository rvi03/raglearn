"""Rules-based query router: exact-figure path vs narrative path.

Deterministic and explainable. A query routes to ``exact`` when it either names a
metric the registry knows (revenue, net income, EPS, …) **or** otherwise reads
like a request for a financial figure (mentions a quantitative line item such as
cash, debt, shares, dividends, capex…). Everything else goes to ``narrative``.

The exact side is intentionally permissive because it *falls back* to narrative
when no fact is found — so a misroute does not produce a wrong answer. With the
text-to-SQL fallback active, a misroute does cost one guarded LLM-SQL attempt
(which then abstains), so the line-item list is kept targeted rather than routing
every question to exact.

This is the ``rules`` adapter; ``llm_classify`` (an LLM labelling the query) is
the heavier alternative behind the same :class:`~finrag.core.interfaces.Router`
interface.
"""

from __future__ import annotations

import re

from finrag.core.registry import registry
from finrag.core.types import Query
from finrag.retrieval.structured_qa import resolve_metric

_EXACT = "exact"
_NARRATIVE = "narrative"

# Quantitative line items that signal a figure question but are not in the metric
# registry — these reach the exact path so the text-to-SQL fallback can try them.
# (Registry metrics like revenue/net income are already caught by resolve_metric.)
_FACT_TERMS = (
    "cash",
    "debt",
    "borrowing",
    "liabilit",  # liability / liabilities
    "asset",
    "equity",
    "dividend",
    "shares outstanding",
    "outstanding shares",
    "capex",
    "capital expenditure",
    "inventor",  # inventory / inventories
    "receivable",
    "payable",
    "ebitda",
    "cash flow",
    "interest expense",
    "goodwill",
    "depreciation",
    "amortization",
    "retained earnings",
    "book value",
    "margin",
    "expenses",
)
_FACT_TERMS_RE = re.compile("|".join(re.escape(t) for t in _FACT_TERMS))


@registry.register("router", "rules")
class RulesRouter:
    """A :class:`~finrag.core.interfaces.Router` that routes by figure intent."""

    def route(self, query: Query) -> list[str]:
        """Return the ordered path(s) to run for a query.

        Returns ``["exact"]`` when the query names a known metric or mentions a
        quantitative line item (the exact path falls back to narrative if no fact
        matches), else ``["narrative"]``.
        """
        if resolve_metric(query.text) is not None or _FACT_TERMS_RE.search(query.text.lower()):
            return [_EXACT]
        return [_NARRATIVE]
