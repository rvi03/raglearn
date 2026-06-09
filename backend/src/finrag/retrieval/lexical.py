"""Lexical (BM25) retrieval: the keyword counterpart of the dense path.

The second retriever, added so fusion has independent rankings to merge. Dense
embeddings capture meaning but miss exact terms — a specific line item, a ticker,
a defined term in a filing — which is precisely what BM25 excels at. Running both
and fusing (see :mod:`finrag.retrieval.fusion`) gets the recall of each.

The index is an in-memory BM25 over the chunk corpus, built once at construction.
Scoring is over the *whole* corpus (so term statistics stay stable), and metadata
filters and the secure-by-default access-tag rule are applied to the scored
results afterwards — mirroring the dense path's Qdrant filtering so the two paths
see the same scoped corpus. A heavier lexical engine (Elasticsearch/OpenSearch)
can replace this behind the same :class:`~finrag.core.interfaces.Retriever`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from finrag.core.registry import registry
from finrag.core.types import Chunk, Query, ScoredChunk

# Tokenizer: lowercase alphanumeric runs. Deliberately simple — BM25 is robust to
# tokenization and the dense path already carries semantic matching.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _passes_filters(chunk: Chunk, filters: dict[str, str]) -> bool:
    """Return whether a chunk's metadata matches every requested filter value."""
    return all(getattr(chunk.metadata, key, None) == value for key, value in filters.items())


def _passes_access(chunk: Chunk, access_tags: Sequence[str]) -> bool:
    """Secure-by-default access: public (no tags) or sharing a caller tag.

    Mirrors the Qdrant access clause so both retrievers enforce the same rule.
    """
    chunk_tags = chunk.metadata.access_tags
    if not chunk_tags:
        return True
    return any(tag in chunk_tags for tag in access_tags)


@registry.register("retriever", "lexical")
class LexicalRetriever:
    """A :class:`~finrag.core.interfaces.Retriever` using an in-memory BM25 index."""

    def __init__(self, *, corpus: Sequence[Chunk]) -> None:
        """Build the BM25 index over a chunk corpus.

        Args:
          corpus: The chunks to index. Empty is allowed (the retriever then
            returns nothing), so startup does not fail on a cold index.
        """
        self._corpus = list(corpus)
        tokenized = [_tokenize(chunk.text) for chunk in self._corpus]
        # Per-chunk token sets define what counts as a lexical hit (term overlap),
        # independent of the BM25 score sign — BM25 gives a *negative* score to a
        # term that appears in most of the corpus, which is still a real match.
        self._token_sets = [set(tokens) for tokens in tokenized]
        # BM25Okapi rejects an empty corpus; guard so a cold index is a no-op.
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def retrieve(self, query: Query, *, top_k: int) -> list[ScoredChunk]:
        """Return up to ``top_k`` chunks by BM25 score, scoped by the query's filters.

        Args:
          query: The user question, its metadata filters, and access tags.
          top_k: Maximum results to return.

        Returns:
          Scored chunks in descending BM25 order, after filtering; only positive
          scores (an actual lexical match) are returned.
        """
        tokens = _tokenize(query.text)
        if self._bm25 is None or not tokens:
            return []
        query_terms = set(tokens)
        scores = self._bm25.get_scores(tokens)
        scored = [
            ScoredChunk(chunk=chunk, score=float(score))
            for chunk, terms, score in zip(self._corpus, self._token_sets, scores, strict=True)
            if query_terms & terms  # a genuine lexical hit, regardless of score sign
            and _passes_filters(chunk, query.filters)
            and _passes_access(chunk, query.access_tags)
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]
