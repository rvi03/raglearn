"""The quality gate: score a chunking and select the best candidate per document.

Each candidate chunking is scored by the intrinsic metrics, combined into one
number as a weighted average over the metrics that apply (a metric returning
``None`` is dropped and the remaining weights carry the average). The candidate
with the highest score wins. This is what lets the router prefer the strategy
that actually chunked a given document best, rather than fixing one strategy.
"""

from __future__ import annotations

import logging

from finrag.core.interfaces.ingestion import Embedder
from finrag.core.types import Chunk, ParsedStructure
from finrag.ingestion.chunking import metrics
from finrag.ingestion.chunking.tokenizer import TokenCounter

logger = logging.getLogger(__name__)

# Equal weight across the four metrics by default; a metric can be excluded by
# setting its weight to zero.
_DEFAULT_WEIGHTS = {"sc": 0.25, "bi": 0.25, "icc": 0.25, "dcc": 0.25}


class QualityGate:
    """Scores candidate chunkings by intrinsic metrics and picks the best."""

    def __init__(
        self,
        embedder: Embedder,
        count_tokens: TokenCounter,
        *,
        weights: dict[str, float] | None = None,
        min_tokens: int = 100,
        max_tokens: int = 512,
        window_tokens: int = 3000,
    ) -> None:
        """Configure the gate with the embedder, token counter, and metric weights."""
        self._embedder = embedder
        self._count = count_tokens
        self._weights = weights or dict(_DEFAULT_WEIGHTS)
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._window_tokens = window_tokens

    def score(self, chunks: list[Chunk], structure: ParsedStructure) -> float:
        """Return the weighted-average intrinsic score of a chunking in ``[0, 1]``."""
        values = {
            "sc": metrics.size_compliance(
                chunks, self._count, min_tokens=self._min_tokens, max_tokens=self._max_tokens
            ),
            "bi": metrics.block_integrity(chunks, structure),
            "icc": metrics.intrachunk_cohesion(chunks, self._embedder),
            "dcc": metrics.document_coherence(
                chunks, self._embedder, self._count, window_tokens=self._window_tokens
            ),
        }
        numerator = sum(
            self._weights.get(name, 0.0) * value
            for name, value in values.items()
            if value is not None
        )
        denominator = sum(
            self._weights.get(name, 0.0) for name, value in values.items() if value is not None
        )
        return numerator / denominator if denominator else 0.0

    def select(
        self, candidates: dict[str, list[Chunk]], structure: ParsedStructure
    ) -> tuple[str, list[Chunk], dict[str, float]]:
        """Score each candidate chunking and return the highest-scoring one.

        Args:
          candidates: Strategy name -> its chunks for this document.
          structure: The source structure the candidates were chunked from.

        Returns:
          The winning strategy name, its chunks, and all candidates' scores.
        """
        scores = {name: self.score(chunks, structure) for name, chunks in candidates.items()}
        best = max(scores, key=lambda name: scores[name])
        return best, candidates[best], scores
