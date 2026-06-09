"""Multi-retriever composition: run several retrievers, fuse, reconcile.

The ``route ↔ fuse`` compose point. When more than one retriever is
configured — dense-hybrid and lexical today, a vectorless tree path later — this
runs each over the query, merges their rankings with RRF, and applies
latest-version reconciliation, presenting one ranking to the reranker above. It is
itself a :class:`~finrag.core.interfaces.Retriever`, so the answer path is
agnostic to whether it is talking to one retriever or many.

Composition over registration: it is assembled from already-built retrievers in
the wiring layer (like the answer service), not selected by config as a leaf
adapter — which leaf retrievers it fuses is a wiring decision, not a single name.
"""

from __future__ import annotations

from collections.abc import Sequence

from finrag.core.interfaces.crosscutting import Tracer
from finrag.core.interfaces.retrieval import Fusion, Retriever
from finrag.core.types import Query, ScoredChunk
from finrag.observability import NullTracer
from finrag.retrieval.reconcile import reconcile_versions


class FusingRetriever:
    """A :class:`~finrag.core.interfaces.Retriever` that fuses several retrievers."""

    def __init__(
        self,
        *,
        retrievers: Sequence[Retriever],
        fusion: Fusion,
        tracer: Tracer | None = None,
    ) -> None:
        """Compose the fusing retriever from its arms and a fusion strategy.

        Args:
          retrievers: The retrievers to run and fuse (at least one).
          fusion: Merges their rankings into one (RRF).
          tracer: Records a span per arm plus the fuse step. Defaults to a no-op.
        """
        if not retrievers:
            raise ValueError("FusingRetriever needs at least one retriever")
        self._retrievers = list(retrievers)
        self._fusion = fusion
        self._tracer = tracer or NullTracer()

    def retrieve(self, query: Query, *, top_k: int) -> list[ScoredChunk]:
        """Run every arm, fuse the rankings, reconcile versions, return ``top_k``.

        Each arm retrieves its own ``top_k`` pool; fusion rewards chunks several
        arms agree on, reconciliation drops superseded versions, and the result is
        capped to ``top_k`` for the reranker above.

        Args:
          query: The user question and its metadata filters.
          top_k: Maximum results to return (also the per-arm pool size).

        Returns:
          The fused, reconciled ranking, at most ``top_k`` chunks.
        """
        rankings: list[list[ScoredChunk]] = []
        for arm in self._retrievers:
            # The arm name rides in ``path`` so the live trace shows each arm
            # distinctly (the agent_step frame surfaces name + path).
            with self._tracer.span("retrieve_arm", path=type(arm).__name__) as span:
                ranking = arm.retrieve(query, top_k=top_k)
                span.set(hits=len(ranking))
            rankings.append(ranking)
        with self._tracer.span("fuse", arms=len(rankings)) as fuse_span:
            fused = reconcile_versions(self._fusion.fuse(rankings))
            fuse_span.set(fused=len(fused))
        return fused[:top_k]
