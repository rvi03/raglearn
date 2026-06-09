"""Tracing integration: the query path emits a nested span tree (hermetic).

Fakes stand in for every stage; a real :class:`InProcessTracer` (with the local
$0 cost model) observes. These pin the span shape the exporters will consume:
the narrative path nests retrieve/rerank/generate under a ``narrative`` span, the
query service roots a ``query`` span with understand/route children, and — the
key cross-service property — the narrative spans nest under the query root
because both services share one tracer.
"""

from __future__ import annotations

from collections.abc import Sequence

from finrag.core.types import (
    Chunk,
    ChunkType,
    DocumentMetadata,
    FactOrigin,
    FinancialFact,
    LLMResponse,
    Market,
    Query,
    ScoredChunk,
    Span,
    Usage,
)
from finrag.cost.local import LocalCostModel
from finrag.observability.tracer import InProcessTracer
from finrag.retrieval.answer import AnswerService, GroundedAnswer
from finrag.retrieval.query_service import QueryService


def _tracer() -> InProcessTracer:
    return InProcessTracer(cost_model=LocalCostModel())


def _names(spans: Sequence[Span]) -> list[str]:
    return [s.name for s in spans]


def _find(root: Span, name: str) -> Span | None:
    if root.name == name:
        return root
    for child in root.children:
        hit = _find(child, name)
        if hit is not None:
            return hit
    return None


def _scored(chunk_id: str, text: str) -> ScoredChunk:
    md = DocumentMetadata(
        collection_id="c1",
        company_name="Mock Corp",
        market=Market.US,
        filing_type="10-K",
        source_doc_id=f"us/{chunk_id}.htm",
    )
    chunk = Chunk(chunk_id=chunk_id, text=text, chunk_type=ChunkType.TEXT, metadata=md)
    return ScoredChunk(chunk=chunk, score=1.0)


class _FakeRetriever:
    def retrieve(self, query: Query, *, top_k: int) -> list[ScoredChunk]:
        return [_scored("cand", "candidate")]


class _FakeReranker:
    def __init__(self, ranked: list[ScoredChunk]) -> None:
        self._ranked = ranked

    def rerank(self, query: Query, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        return self._ranked


class _FakeLLM:
    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            text="Net sales were $123B [1].",
            usage=Usage(tokens_in=12, tokens_out=8),
            model="qwen2.5",
        )


def _answer_service(tracer: InProcessTracer, ranked: list[ScoredChunk]) -> AnswerService:
    return AnswerService(
        retriever=_FakeRetriever(),  # type: ignore[arg-type]
        reranker=_FakeReranker(ranked),  # type: ignore[arg-type]
        llm=_FakeLLM(),  # type: ignore[arg-type]
        tracer=tracer,
    )


def test_narrative_path_emits_retrieve_rerank_generate_spans() -> None:
    tracer = _tracer()
    service = _answer_service(tracer, [_scored("a", "Net sales were $123B.")])

    service.answer(Query(text="What were net sales?"))

    root = tracer.last_trace
    assert root is not None
    assert root.name == "narrative"
    assert _names(root.children) == ["retrieve", "rerank", "generate"]
    # The rerank span records how many chunks survived as evidence.
    rerank = _find(root, "rerank")
    assert rerank is not None
    assert rerank.attributes.get("kept") == 1
    # The generate span carries the LLM call's usage and model.
    generate = _find(root, "generate")
    assert generate is not None
    assert generate.usage == Usage(tokens_in=12, tokens_out=8)
    assert generate.model == "qwen2.5"


def test_empty_retrieval_traces_retrieve_and_rerank_but_not_generate() -> None:
    tracer = _tracer()
    service = _answer_service(tracer, [])  # reranker yields nothing → abstain

    result = service.answer(Query(text="anything"))

    assert result.answered is False
    root = tracer.last_trace
    assert root is not None
    assert _find(root, "retrieve") is not None
    assert _find(root, "rerank") is not None
    assert _find(root, "generate") is None  # abstained before the LLM


# --- QueryService roots the tree -----------------------------------------------


class _Router:
    def __init__(self, paths: list[str]) -> None:
        self._paths = paths

    def route(self, query: Query) -> list[str]:
        return self._paths


class _QA:
    def __init__(self, facts: list[FinancialFact]) -> None:
        self._facts = facts

    def answer(self, query: Query) -> list[FinancialFact]:
        return self._facts


class _Store:
    def __init__(self, found: tuple[str, str] | None) -> None:
        self._found = found

    def find_collection(self, text: str) -> tuple[str, str] | None:
        return self._found


def _fact() -> FinancialFact:
    return FinancialFact(
        fact_id="nw-rev-24",
        filing_id="nw-10k",
        concept="us-gaap:Revenues",
        value=4250.0,
        unit="USD",
        period="2023-10-01/2024-09-30",
        dimension=None,
        origin=FactOrigin.XBRL,
    )


def test_exact_path_roots_a_query_span_with_understand_route_exact() -> None:
    tracer = _tracer()

    class _UnusedNarrative:
        def answer(self, query: Query) -> GroundedAnswer:  # pragma: no cover - not taken
            raise AssertionError("narrative should not run on a satisfied exact route")

    service = QueryService(
        router=_Router(["exact"]),  # type: ignore[arg-type]
        structured_qa=_QA([_fact()]),  # type: ignore[arg-type]
        narrative=_UnusedNarrative(),  # type: ignore[arg-type]
        store=_Store(("us-cik-100", "Northwind Trading Co.")),  # type: ignore[arg-type]
        tracer=tracer,
    )

    service.answer(Query(text="Northwind total revenue FY2024"))

    root = tracer.last_trace
    assert root is not None
    assert root.name == "query"
    assert _names(root.children) == ["understand", "route", "exact"]
    assert root.attributes.get("path") == "exact"
    route = _find(root, "route")
    assert route is not None
    assert route.attributes.get("path") == "exact"


def test_narrative_spans_nest_under_the_query_root_via_shared_tracer() -> None:
    tracer = _tracer()
    narrative = _answer_service(tracer, [_scored("a", "Net sales were $123B.")])
    service = QueryService(
        router=_Router(["narrative"]),  # type: ignore[arg-type]
        structured_qa=_QA([]),  # type: ignore[arg-type]
        narrative=narrative,
        store=_Store(None),  # type: ignore[arg-type]
        tracer=tracer,
    )

    service.answer(Query(text="What are the risks?"))

    root = tracer.last_trace
    assert root is not None
    assert root.name == "query"
    assert root.attributes.get("path") == "narrative"
    # The whole narrative subtree hangs under the query root — one trace, not two.
    narrative_span = _find(root, "narrative")
    assert narrative_span is not None
    assert _find(narrative_span, "generate") is not None
