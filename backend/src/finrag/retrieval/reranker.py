"""bge-reranker-v2-m3 cross-encoder reranking (in-process, CPU).

Re-scores retrieval candidates against the query with a cross-encoder. Unlike the
bi-encoder retrieval that produced the candidates (query and chunk embedded
separately), a cross-encoder reads the query and a chunk *together*, so it ranks
relevance far more precisely — at the cost of one forward pass per candidate,
which is why it runs only over the shortlist retrieval returns.

bge-reranker-v2-m3 is a sequence-classification model: each ``(query, chunk)``
pair yields a single relevance logit, squashed to ``[0, 1]`` with a sigmoid. We
run it through ``transformers`` directly (``AutoModelForSequenceClassification``)
rather than FlagEmbedding's reranker wrapper, which calls a tokenizer method that
recent ``transformers`` no longer exposes.

Same discipline as the bge-m3 embedder: the model is heavy, so it loads lazily
and is cached on first ``rerank`` (never at import or in tests that inject a
fake), and it is pinned to CPU for deterministic, machine-independent scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from finrag.core.registry import registry
from finrag.core.types import Query, ScoredChunk

_BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def _load_model(model_name: str, device: str) -> Any:
    """Load and cache the reranker's tokenizer + model (lazy: imports torch stack)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def _compute_scores(
    bundle: Any, pairs: list[list[str]], *, batch_size: int, max_length: int, normalize: bool
) -> list[float]:
    """Score each ``(query, chunk)`` pair with the cross-encoder, in batches.

    Args:
      bundle: The ``(tokenizer, model)`` pair from :func:`_load_model`.
      pairs: ``[query, chunk_text]`` pairs to score.
      batch_size: Pairs per forward pass.
      max_length: Token cap per pair.
      normalize: Squash logits to ``[0, 1]`` with a sigmoid.

    Returns:
      One relevance score per pair, in input order.
    """
    import torch

    tokenizer, model = bundle
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(model.device)
            logits = model(**inputs).logits.view(-1).float()
            if normalize:
                logits = torch.sigmoid(logits)
            scores.extend(logits.tolist())
    return scores


@registry.register("reranker", "bge_reranker")
class BgeReranker:
    """A :class:`~finrag.core.interfaces.Reranker` backed by bge-reranker-v2-m3."""

    def __init__(
        self,
        *,
        model_name: str = _BGE_RERANKER_MODEL,
        device: str = "cpu",
        batch_size: int = 12,
        max_length: int = 512,
        normalize: bool = True,
    ) -> None:
        """Configure the reranker; the model itself loads lazily on first rerank.

        Args:
          model_name: Hugging Face id of the reranker model to load.
          device: Device to run on; ``"cpu"`` by default for deterministic output.
          batch_size: Query/chunk pairs scored per forward pass.
          max_length: Token cap per pair (bge-reranker's documented default).
          normalize: Squash relevance logits to ``[0, 1]`` with a sigmoid.
        """
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._normalize = normalize

    def rerank(self, query: Query, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Return the candidates reordered by cross-encoder relevance, best first.

        Args:
          query: The user question.
          chunks: The retrieval candidates to re-score (their incoming scores are
            from the bi-encoder and are replaced by the cross-encoder score).

        Returns:
          The same chunks, re-scored and sorted by descending relevance. Empty
          input yields an empty list without loading the model.
        """
        if not chunks:
            return []
        bundle = _load_model(self._model_name, self._device)
        pairs = [[query.text, scored.chunk.text] for scored in chunks]
        scores = _compute_scores(
            bundle,
            pairs,
            batch_size=self._batch_size,
            max_length=self._max_length,
            normalize=self._normalize,
        )
        reranked = [
            ScoredChunk(chunk=scored.chunk, score=float(score))
            for scored, score in zip(chunks, scores, strict=True)
        ]
        reranked.sort(key=lambda scored: scored.score, reverse=True)
        return reranked
