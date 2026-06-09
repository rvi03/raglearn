"""Run the synthetic retrieval evaluation against live services.

Wires the real embedder, vector store, retriever, reranker, and LLM; ingests the
synthetic corpus into a *dedicated* eval collection (so it never touches the real
index); runs the golden set; and logs the per-case scores plus the leaderboard
summary. Needs Qdrant + Ollama running and the bge-m3 / bge-reranker models
available — it is a local out-of-band run, not part of the hermetic test suite.

    python -m finrag.eval
"""

from __future__ import annotations

from finrag.core.bootstrap import load_adapters
from finrag.core.config import load_settings
from finrag.core.logging import configure_logging, get_logger
from finrag.core.registry import registry
from finrag.core.wiring import resolve_adapter
from finrag.eval.leaderboard import Leaderboard
from finrag.eval.runner import ingest_corpus, ingest_facts, run_eval, summarize
from finrag.eval.synthetic import synthetic_corpus, synthetic_facts
from finrag.retrieval.answer import AnswerService
from finrag.retrieval.composite_qa import build_structured_qa
from finrag.retrieval.query_service import QueryService

logger = get_logger(__name__)

# A throwaway collection for eval, kept apart from the real chunk index.
_EVAL_COLLECTION = "finrag_eval"
# An ephemeral in-process DuckDB for the synthetic facts — never the real store.
_EVAL_DB = ":memory:"


def _fmt(value: float, answerable: bool) -> str:
    """Format a quality metric, blanking it for negative cases (nothing to expect)."""
    return f"{value:.2f}" if answerable else "  — "


def main() -> None:
    """Ingest the synthetic corpus, run the golden set, and report the scores."""
    settings = load_settings()
    configure_logging(settings.log_level)
    load_adapters()

    embedder = resolve_adapter(settings, "embedder")
    vector_store = resolve_adapter(
        settings, "vector_store", url=settings.services.qdrant_url, collection=_EVAL_COLLECTION
    )
    retriever = resolve_adapter(settings, "retriever", embedder=embedder, vector_store=vector_store)
    reranker = resolve_adapter(settings, "reranker")
    llm = resolve_adapter(settings, "llm_backend", url=settings.services.ollama_url)
    harness = resolve_adapter(settings, "harness", llm=llm)
    narrative = AnswerService(retriever=retriever, reranker=reranker, llm=llm, harness=harness)

    # The exact path reads from an isolated in-memory store seeded with mock facts.
    # Pinned to DuckDB (not the active adapter) so the eval harness stays hermetic
    # regardless of the deployed structured store.
    structured_store = registry.create("structured_store", "duckdb", path=_EVAL_DB)
    structured_qa = build_structured_qa(settings, store=structured_store, llm=llm)
    router = resolve_adapter(settings, "router")
    query_transform = resolve_adapter(settings, "query_transform", llm=llm)
    service = QueryService(
        router=router,
        structured_qa=structured_qa,
        narrative=narrative,
        store=structured_store,
        query_transform=query_transform,
    )

    chunks, cases = synthetic_corpus()
    extractions = synthetic_facts()
    logger.info(
        "ingesting %d chunks into %r and %d synthetic filings into the eval store",
        len(chunks),
        _EVAL_COLLECTION,
        len(extractions),
    )
    ingest_corpus(chunks, embedder, vector_store)
    ingest_facts(extractions, structured_store)

    logger.info("running %d golden cases", len(cases))
    results = run_eval(cases, service)
    for result in results:
        # Quality metrics are meaningless for a negative case (nothing to expect),
        # so show "—" there rather than a vacuous 1.00.
        ans = result.answerable
        grounding = f"{result.grounding_confidence:.2f}" if result.grounding_confidence else "  — "
        logger.info(
            "%-15s answerable=%-5s answered=%-5s abst_ok=%-5s "
            "recall=%s cite=%s numeric=%s ground=%s",
            result.case_id,
            result.answerable,
            result.answered,
            result.abstention_correct,
            _fmt(result.context_recall, ans),
            _fmt(result.citation_correctness, ans),
            _fmt(result.numeric_match, ans),
            grounding,
        )

    scores = summarize(results)
    config = (
        f"retriever={settings.adapters['retriever'].active}"
        f"/reranker={settings.adapters['reranker'].active}"
    )
    board = Leaderboard()
    board.add(config, scores)
    logger.info("SUMMARY %s -> %s", config, scores)


if __name__ == "__main__":
    main()
