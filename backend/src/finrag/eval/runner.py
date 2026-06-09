"""The retrieval evaluation runner.

Ties the synthetic (or any) corpus and golden set to the live answer path and
turns the result into scores. Two steps:

  1. :func:`ingest_corpus` — embed the corpus chunks and upsert them into the
     vector store, so retrieval can find them (the eval's setup).
  2. :func:`run_eval` — for each golden case, ask the answer service and score
     its output against the case, then :func:`summarize` averages the per-case
     scores into a row for the leaderboard.

Scoring is deliberately strict-but-simple (the full LLM-judge faithfulness loop
is the Generation vertical's job):

  * **context_recall** — were the expected chunks among the evidence retrieved?
  * **citation_correctness** — did the answer *cite* an expected chunk (not just
    retrieve it)?
  * **numeric_match** — does the answer state the expected figures?
  * **answer_rate** — did the system answer rather than abstain?
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from finrag.core.interfaces.ingestion import Embedder
from finrag.core.interfaces.storage import StructuredStore, VectorStore
from finrag.core.types import Chunk, Query, XbrlExtraction
from finrag.eval import metrics
from finrag.eval.golden import GoldenCase
from finrag.retrieval.answer import GroundedAnswer
from finrag.retrieval.query_service import QueryService


class CaseResult(BaseModel):
    """The scores for one golden case."""

    case_id: str
    answerable: bool
    answered: bool
    abstention_correct: bool  # answered when it should, abstained when it should
    context_recall: float
    citation_correctness: float
    numeric_match: float
    grounding_confidence: float | None = None  # harness verdict, None if unverified


def ingest_corpus(chunks: Sequence[Chunk], embedder: Embedder, vector_store: VectorStore) -> None:
    """Embed the corpus chunks and upsert them so retrieval can find them.

    Args:
      chunks: The corpus to index.
      embedder: Embeds the chunk texts (same model retrieval embeds queries with).
      vector_store: Receives the chunks and their embeddings.
    """
    vectors = embedder.embed([chunk.text for chunk in chunks])
    vector_store.upsert(chunks, vectors)


def ingest_facts(extractions: Sequence[XbrlExtraction], store: StructuredStore) -> None:
    """Write synthetic structured filings so the exact path can read them.

    Args:
      extractions: The synthetic filings (collection + filing + facts).
      store: The structured store to write into (the exact path's read source).
    """
    for extraction in extractions:
        store.write(extraction)


def evaluate_case(case: GoldenCase, answer: GroundedAnswer) -> CaseResult:
    """Score one answer against its golden case.

    Args:
      case: The expected chunks and figures for the query.
      answer: What the system produced for the query.

    Returns:
      The per-case scores.
    """
    source_chunk = {source.id: source.chunk_id for source in answer.sources}
    retrieved_ids = list(source_chunk.values())
    cited_ids = [source_chunk[c.id] for c in answer.citations if c.id in source_chunk]
    return CaseResult(
        case_id=case.id,
        answerable=case.answerable,
        answered=answer.answered,
        abstention_correct=answer.answered == case.answerable,
        context_recall=metrics.recall(retrieved_ids, case.expected_chunk_ids),
        citation_correctness=metrics.recall(cited_ids, case.expected_chunk_ids),
        numeric_match=metrics.numeric_match(answer.answer, case.expected_values),
        grounding_confidence=answer.grounding_confidence,
    )


def run_eval(cases: Sequence[GoldenCase], query_service: QueryService) -> list[CaseResult]:
    """Run every golden case through the query service and score each one.

    The query service routes each case to the exact or narrative path, so one run
    scores both kinds of question.

    Args:
      cases: The golden set.
      query_service: The routed query service under evaluation.

    Returns:
      One :class:`CaseResult` per case, in input order.
    """
    return [evaluate_case(case, query_service.answer(Query(text=case.query))) for case in cases]


def summarize(results: Sequence[CaseResult]) -> dict[str, float]:
    """Average the per-case scores into a single metric row (empty input -> ``{}``).

    ``abstention_accuracy`` spans every case (did the system answer/abstain when it
    should?). The quality metrics — recall, citation, numeric — are averaged over
    the **answerable** cases only, since they are meaningless for a case whose
    correct outcome is "no answer" (there is no chunk or figure to expect).
    """
    if not results:
        return {}
    answerable = [r for r in results if r.answerable]
    summary = {"abstention_accuracy": sum(r.abstention_correct for r in results) / len(results)}
    if answerable:
        m = len(answerable)
        summary["context_recall"] = sum(r.context_recall for r in answerable) / m
        summary["citation_correctness"] = sum(r.citation_correctness for r in answerable) / m
        summary["numeric_match"] = sum(r.numeric_match for r in answerable) / m
    graded = [r for r in answerable if r.grounding_confidence is not None]
    if graded:
        summary["grounding"] = sum(r.grounding_confidence or 0.0 for r in graded) / len(graded)
    return summary
