"""Cross-cutting interfaces.

Concerns that ride alongside every stage rather than sitting in the pipeline,
such as cost accounting.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from raglearn.core.types import CostBreakdown, Usage


@runtime_checkable
class CostModel(Protocol):
    """Prices token usage for a model.

    Implementations price from a serverless sheet, a self-hosted ``$/hr``
    amortization, or a flat zero for local dev.
    """

    def price(self, usage: Usage, model: str) -> CostBreakdown:
        """Return the cost breakdown for a unit of usage on a model."""
        ...
