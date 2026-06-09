"""Chain structured-QA stages so the first to find facts wins.

Lets the deterministic metric registry stay authoritative for the metrics it
knows while a text-to-SQL stage answers off-registry fact questions: the order is
``[metric_registry, text_to_sql]``, so the registry is always tried first and the
LLM path only runs when it would otherwise be no answer at all.
"""

from __future__ import annotations

from finrag.core.config import Settings
from finrag.core.interfaces.generation import LLMBackend
from finrag.core.interfaces.retrieval import StructuredQA
from finrag.core.registry import registry
from finrag.core.types import FinancialFact, Query


class FallbackStructuredQA:
    """A :class:`~finrag.core.interfaces.StructuredQA` that tries stages in order."""

    def __init__(self, stages: list[StructuredQA]) -> None:
        """Bind the chain of QA stages, tried in order until one returns facts."""
        self._stages = stages

    def answer(self, query: Query) -> list[FinancialFact]:
        """Return the first stage's non-empty facts, or ``[]`` if none answer."""
        for stage in self._stages:
            facts = stage.answer(query)
            if facts:
                return facts
        return []


def build_structured_qa(settings: Settings, *, store: object, llm: LLMBackend) -> StructuredQA:
    """Build the exact-path QA from config: metric registry alone or chained with SQL.

    The metric registry is always the authoritative first stage. When
    ``structured_qa.active = text_to_sql``, a guarded LLM text-to-SQL stage is
    chained after it as a fallback for questions the registry cannot map.

    Args:
      settings: Loaded application settings.
      store: The structured store the QA stages read through.
      llm: The backend the text-to-SQL stage generates with (used only when active).

    Returns:
      The composed :class:`~finrag.core.interfaces.StructuredQA`.
    """
    metric_qa = registry.create("structured_qa", "metric_registry", store=store)
    if settings.adapters["structured_qa"].active == "text_to_sql":
        sql_qa = registry.create("structured_qa", "text_to_sql", llm=llm, store=store)
        return FallbackStructuredQA([metric_qa, sql_qa])
    return metric_qa  # type: ignore[no-any-return]
