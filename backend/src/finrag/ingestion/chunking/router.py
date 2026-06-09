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

from finrag.core.interfaces.ingestion import Chunker, Embedder
from finrag.core.types import Chunk, DocumentMetadata, ParsedStructure
from finrag.ingestion.chunking.fixed import FixedChunker
from finrag.ingestion.chunking.gate import QualityGate
from finrag.ingestion.chunking.semantic import SemanticChunker
from finrag.ingestion.chunking.structure import StructureChunker
from finrag.ingestion.chunking.tokenizer import TokenCounter

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
        """Yield the winning chunking for a document (see :meth:`chunk_with_decision`)."""
        chunks, _ = self.chunk_with_decision(structure, metadata)
        yield from chunks

    def chunk_with_decision(
        self, structure: ParsedStructure, metadata: DocumentMetadata
    ) -> tuple[list[Chunk], dict[str, object]]:
        """Return the winning chunking *and* the routing decision behind it.

        The decision (chosen strategy + per-strategy scores) is what the monitor
        DAG surfaces, so an operator can see *which* strategy won and by how much —
        not just that chunking happened.

        With a pinned strategy, runs only that one (no scoring). Otherwise runs
        every strategy, scores each non-empty result with the gate, and returns the
        best. Returns ``([], {...})`` if every strategy produces no chunks.
        """
        if self._pinned is not None:
            chunks = list(self._strategies[self._pinned].chunk(structure, metadata))
            return chunks, {"strategy": self._pinned, "pinned": True, "scores": {}}

        candidates = {
            name: list(strategy.chunk(structure, metadata))
            for name, strategy in self._strategies.items()
        }
        candidates = {name: chunks for name, chunks in candidates.items() if chunks}
        if not candidates:
            logger.info("chunk router: no strategy produced chunks for %s", metadata.source_doc_id)
            return [], {"strategy": None, "scores": {}}

        best, chunks, scores = self._gate.select(candidates, structure)
        rounded = {name: round(score, 3) for name, score in scores.items()}
        counts = {name: len(c) for name, c in candidates.items()}
        logger.info(
            "chunk router picked %s for %s (scores=%s)", best, metadata.source_doc_id, rounded
        )
        return list(chunks), {"strategy": best, "scores": rounded, "counts": counts}


def build_chunk_router(
    embedder: Embedder, count_tokens: TokenCounter, *, pinned: str | None = None
) -> ChunkRouter:
    """Build the adaptive chunk router with the three built-in strategies.

    The strategy names match the ``chunker`` capability matrix
    (``structure_aware`` / ``semantic`` / ``fixed``). The semantic strategy and
    the gate need the embedder; all three need the token counter.

    Args:
      embedder: Embedder for the semantic strategy and the quality gate.
      count_tokens: Token counter aligned to the embedder.
      pinned: If set, always use that strategy and skip gate scoring.
    """
    strategies: dict[str, Chunker] = {
        "structure_aware": StructureChunker(count_tokens),
        "semantic": SemanticChunker(embedder, count_tokens),
        "fixed": FixedChunker(count_tokens),
    }
    gate = QualityGate(embedder, count_tokens)
    return ChunkRouter(strategies, gate, pinned=pinned)
