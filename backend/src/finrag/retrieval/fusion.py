"""Reciprocal Rank Fusion (RRF) over multiple retriever rankings.

When more than one retriever answers a query — the dense-hybrid path and a
lexical (BM25) path today, a vectorless tree path later — their rankings must be
merged into one. RRF is the standard, score-agnostic way to do this: each result
contributes ``1 / (k + rank)`` from every list it appears in, and the
contributions sum. It needs no score calibration across retrievers (a cosine
similarity and a BM25 score are not comparable), only their *order*, which makes
it robust and is why it is the default fusion for hybrid search.

Cross-path dedup falls out for free: the same chunk retrieved by two paths has
its contributions added, so agreement between retrievers is rewarded.
"""

from __future__ import annotations

from collections.abc import Sequence

from finrag.core.registry import registry
from finrag.core.types import Chunk, ScoredChunk

# The RRF damping constant. 60 is the value from the original Cormack et al. paper
# and the de-facto default; it flattens the contribution of deep ranks so the top
# of each list dominates without any single list being able to force a winner.
_DEFAULT_K = 60


@registry.register("fusion", "rrf")
class RrfFusion:
    """A :class:`~finrag.core.interfaces.Fusion` using reciprocal rank fusion."""

    def __init__(self, *, k: int = _DEFAULT_K) -> None:
        """Bind the fusion to its damping constant.

        Args:
          k: The RRF constant; larger flattens rank weighting. Defaults to 60.
        """
        self._k = k

    def fuse(self, results: Sequence[Sequence[ScoredChunk]]) -> list[ScoredChunk]:
        """Merge ranked result lists into one deduplicated RRF ranking.

        Args:
          results: One ranking per retriever, each already in descending relevance.

        Returns:
          A single ranking, descending by fused score, with each chunk appearing
          once (its score is the sum of its RRF contributions across the lists).
        """
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for ranking in results:
            for rank, scored in enumerate(ranking):
                chunk_id = scored.chunk.chunk_id
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self._k + rank + 1)
                chunks.setdefault(chunk_id, scored.chunk)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [ScoredChunk(chunk=chunks[chunk_id], score=score) for chunk_id, score in ordered]
