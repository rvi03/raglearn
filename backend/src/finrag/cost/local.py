"""Local cost model: captures tokens, prices everything at zero.

Local dev runs on Ollama with no per-token charge, but token counts still feed
the comparison leaderboard, so the DAG and chat render ``$0.00`` rather than
blank.
"""

from __future__ import annotations

from finrag.core.registry import registry
from finrag.core.types import CostBreakdown, Usage


@registry.register("cost_model", "local")
class LocalCostModel:
    """A :class:`~finrag.core.interfaces.CostModel` that always prices at $0."""

    def price(self, usage: Usage, model: str) -> CostBreakdown:
        """Return a zero-dollar breakdown that preserves the token counts.

        Args:
          usage: Token counts for the call.
          model: The model name, echoed back for reporting.

        Returns:
          A :class:`CostBreakdown` with ``usd=0.0``.
        """
        return CostBreakdown(
            model=model,
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
            usd=0.0,
        )
