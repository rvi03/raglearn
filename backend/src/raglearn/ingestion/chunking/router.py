"""The adaptive chunk router: run each strategy, keep the best per document.

Runs every chunking strategy on the document, scores each result with the
quality gate, and emits the winning chunking. Different documents win with
different strategies -- a well-structured filing favours the structure strategy,
a flat one favours semantic -- which is the point of routing over fixing one
strategy. A single strategy can be pinned (for the evaluation leaderboard or a
cheap mode), in which case the gate is bypassed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from raglearn.core.interfaces.ingestion import Chunker
from raglearn.core.types import Chunk, DocumentMetadata, ParsedStructure
from raglearn.ingestion.chunking.gate import QualityGate

logger = logging.getLogger(__name__)


class ChunkRouter:
    """Selects, per document, the best-scoring chunking among several strategies."""

    def __init__(
        self,
        strategies: dict[str, Chunker],
        gate: QualityGate,
        *,
        pinned: str | None = None,
    ) -> None:
        """Bind the named strategies and the gate that scores them.

        Args:
          strategies: Strategy name -> chunker.
          gate: The quality gate that scores and selects candidate chunkings.
          pinned: If set, always use this strategy and skip scoring.

        Raises:
          ValueError: If ``pinned`` names a strategy that is not registered.
        """
        if pinned is not None and pinned not in strategies:
            raise ValueError(f"pinned strategy not registered: {pinned!r}")
        self._strategies = dict(strategies)
        self._gate = gate
        self._pinned = pinned

    def chunk(self, structure: ParsedStructure, metadata: DocumentMetadata) -> Iterator[Chunk]:
        """Yield the winning chunking for a document.

        With a pinned strategy, runs only that one. Otherwise runs every
        strategy, scores each non-empty result with the gate, and yields the
        best. Yields nothing if every strategy produces no chunks.
        """
        if self._pinned is not None:
            yield from self._strategies[self._pinned].chunk(structure, metadata)
            return

        candidates = {
            name: list(strategy.chunk(structure, metadata))
            for name, strategy in self._strategies.items()
        }
        candidates = {name: chunks for name, chunks in candidates.items() if chunks}
        if not candidates:
            logger.info("chunk router: no strategy produced chunks for %s", metadata.source_doc_id)
            return

        best, chunks, scores = self._gate.select(candidates, structure)
        logger.info(
            "chunk router picked %s for %s (scores=%s)",
            best,
            metadata.source_doc_id,
            {name: round(score, 3) for name, score in scores.items()},
        )
        yield from chunks
