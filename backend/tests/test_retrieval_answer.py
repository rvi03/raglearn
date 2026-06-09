"""Tests for the minimal grounded answer service (hermetic: fakes throughout).

No retrieval, reranking, or LLM runs. Fakes stand in for each stage so these
tests pin the orchestration: candidate_k/top_k are honoured, the prompt is
grounded and numbered, the model's ``[n]`` markers map back to source
provenance (and only to real sources), evidence cards are built, and an empty
retrieval abstains without ever calling the LLM.
"""

from __future__ import annotations

from collections.abc import Sequence

from finrag.core.types import (
    Chunk,
    ChunkType,
    DocumentMetadata,
    LLMResponse,
    Market,
    Query,
    ScoredChunk,
    Usage,
)
from finrag.retrieval.answer import AnswerService, _build_prompt, _citations


def _scored(
    chunk_id: str,
    text: str,
    *,
    page: int | None = None,
    section: str | None = None,
    period: str | None = None,
) -> ScoredChunk:
    md = DocumentMetadata(
        collection_id="c1",
        company_name="Mock Corp",
        market=Market.US,
        filing_type="10-K",
        fiscal_period=period,
        section=section,
        source_doc_id=f"us/{chunk_id}.htm",
        page=page,
    )
    chunk = Chunk(chunk_id=chunk_id, text=text, chunk_type=ChunkType.TEXT, metadata=md)
    return ScoredChunk(chunk=chunk, score=1.0)


class _FakeRetriever:
    def __init__(self, results: list[ScoredChunk]) -> None:
        self.results = results
        self.top_k: int | None = None

    def retrieve(self, query: Query, *, top_k: int) -> list[ScoredChunk]:
        self.top_k = top_k
        return self.results


class _FakeReranker:
    """Returns a fixed ranking, ignoring its input (ordering is tested elsewhere)."""

    def __init__(self, ranked: list[ScoredChunk]) -> None:
        self.ranked = ranked

    def rerank(self, query: Query, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        return self.ranked


class _FakeLLM:
    def __init__(self, text: str, usage: Usage | None = None) -> None:
        self.text = text
        self.usage = usage or Usage(tokens_in=3, tokens_out=4)
        self.prompt: str | None = None

    def generate(self, prompt: str) -> LLMResponse:
        self.prompt = prompt
        return LLMResponse(text=self.text, usage=self.usage)


def _service(
    *,
    ranked: list[ScoredChunk],
    answer: str,
    candidate_k: int = 20,
    top_k: int = 5,
    llm: _FakeLLM | None = None,
) -> tuple[AnswerService, _FakeRetriever, _FakeLLM]:
    retriever = _FakeRetriever([_scored("cand", "candidate")])
    backend = llm or _FakeLLM(answer)
    service = AnswerService(
        retriever=retriever,
        reranker=_FakeReranker(ranked),
        llm=backend,
        candidate_k=candidate_k,
        top_k=top_k,
    )
    return service, retriever, backend


def test_assembles_answer_citations_and_sources() -> None:
    ranked = [
        _scored("a", "Net sales were $123B.", page=31, section="Item 7", period="FY2023"),
        _scored("b", "Revenue grew across segments.", page=40),
    ]
    service, _retriever, llm = _service(
        ranked=ranked, answer="Net sales were $123B [1]; it grew [2]."
    )

    result = service.answer(Query(text="What were net sales?"))

    assert result.answer == "Net sales were $123B [1]; it grew [2]."
    assert result.answered is True
    assert {c.id for c in result.citations} == {1, 2}
    first = next(c for c in result.citations if c.id == 1)
    assert first.source_doc_id == "us/a.htm"
    assert first.page == 31
    assert first.section == "Item 7"
    assert first.period == "FY2023"
    assert [s.id for s in result.sources] == [1, 2]
    assert result.sources[0].title == "Mock Corp · 10-K · FY2023 · Item 7"
    assert result.sources[0].url == "/sources/us/a.htm#p31"
    assert result.usage == llm.usage


def test_abstains_without_calling_llm_when_no_evidence() -> None:
    service, _retriever, llm = _service(ranked=[], answer="should not be used")

    result = service.answer(Query(text="anything"))

    assert result.answered is False
    assert result.citations == []
    assert result.sources == []
    assert llm.prompt is None  # the LLM is never invoked
    assert "cannot answer" in result.answer.lower()


def test_model_refusal_marks_unanswered_and_strips_citations() -> None:
    # Evidence is retrieved, but the model judges it insufficient and refuses —
    # while sloppily appending citation markers. Those markers must be ignored.
    ranked = [_scored("a", "x"), _scored("b", "y")]
    service, _retriever, _llm = _service(
        ranked=ranked, answer="I cannot answer from the provided documents. [1] [2]"
    )

    result = service.answer(Query(text="q"))

    assert result.answered is False
    assert result.citations == []  # a refusal cites nothing, despite the markers
    assert len(result.sources) == 2  # evidence kept so retrieval stays measurable
    assert "cannot answer" in result.answer.lower()


def test_only_cited_sources_become_citations() -> None:
    ranked = [_scored("a", "x"), _scored("b", "y"), _scored("c", "z")]
    service, _retriever, _llm = _service(ranked=ranked, answer="Only the middle one matters [2].")

    result = service.answer(Query(text="q"))

    assert [c.id for c in result.citations] == [2]
    assert [s.id for s in result.sources] == [1, 2, 3]  # all evidence still surfaced


def test_hallucinated_citation_ids_are_dropped() -> None:
    ranked = [_scored("a", "x"), _scored("b", "y")]
    service, _retriever, _llm = _service(ranked=ranked, answer="See [5].")  # only 2 sources exist

    result = service.answer(Query(text="q"))

    assert result.citations == []


def test_candidate_k_drives_retrieval_and_top_k_caps_evidence() -> None:
    ranked = [_scored(str(i), "t") for i in range(4)]
    service, retriever, _llm = _service(ranked=ranked, answer="a [1]", candidate_k=11, top_k=2)

    result = service.answer(Query(text="q"))

    assert retriever.top_k == 11  # retrieve broad
    assert len(result.sources) == 2  # then keep only top_k as evidence


def test_prompt_is_grounded_and_numbered() -> None:
    ranked = [_scored("a", "Net sales were $123B.", period="FY2023")]

    prompt = _build_prompt(Query(text="What were net sales?"), ranked)

    assert "ONLY" in prompt  # evidence-only instruction
    assert "[1]" in prompt and "Mock Corp · 10-K · FY2023" in prompt  # numbered source
    assert "Net sales were $123B." in prompt  # the data block
    assert "QUESTION: What were net sales?" in prompt


def test_citations_helper_ignores_unreferenced_sources() -> None:
    ranked = [_scored("a", "x"), _scored("b", "y")]

    assert _citations("no markers here", ranked) == []
    assert [c.id for c in _citations("just [1]", ranked)] == [1]


# --- harness (verify → regenerate) loop ---------------------------------------


class _SeqLLM:
    """Returns a queued sequence of answers, one per generate call."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.calls = 0

    def generate(self, prompt: str) -> LLMResponse:
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        return LLMResponse(text=text, usage=Usage(tokens_in=1, tokens_out=1))


class _FakeHarness:
    """Scores each draft with a queued confidence; optionally drops citation ids."""

    def __init__(self, confidences: list[float], *, drop: set[int] | None = None) -> None:
        self._confidences = confidences
        self._drop = drop or set()
        self.calls = 0

    def apply(self, draft: object, evidence: object) -> object:
        confidence = self._confidences[min(self.calls, len(self._confidences) - 1)]
        self.calls += 1
        kept = [c for c in draft.citations if c.id not in self._drop]  # type: ignore[attr-defined]
        return draft.model_copy(  # type: ignore[attr-defined]
            update={"grounding_confidence": confidence, "citations": kept}
        )


def _harness_service(
    *, ranked: list[ScoredChunk], llm: object, harness: object, max_attempts: int = 2
) -> AnswerService:
    return AnswerService(
        retriever=_FakeRetriever([_scored("cand", "candidate")]),
        reranker=_FakeReranker(ranked),
        llm=llm,  # type: ignore[arg-type]
        harness=harness,  # type: ignore[arg-type]
        max_attempts=max_attempts,
        grounding_threshold=0.6,
    )


def test_harness_accepts_grounded_first_attempt() -> None:
    llm = _SeqLLM(["Net sales were $123B [1]."])
    harness = _FakeHarness([0.9])
    service = _harness_service(ranked=[_scored("a", "x")], llm=llm, harness=harness)

    result = service.answer(Query(text="q"))

    assert result.answered is True
    assert result.grounding_confidence == 0.9
    assert llm.calls == 1  # cleared the bar first try, no regeneration


def test_harness_regenerates_until_grounded() -> None:
    llm = _SeqLLM(["weak [1]", "better [1]"])
    harness = _FakeHarness([0.3, 0.8])  # first below threshold, second clears it
    service = _harness_service(ranked=[_scored("a", "x")], llm=llm, harness=harness)

    result = service.answer(Query(text="q"))

    assert result.answered is True
    assert result.answer == "better [1]"
    assert result.grounding_confidence == 0.8
    assert llm.calls == 2  # regenerated once


def test_harness_abstains_when_never_grounded() -> None:
    llm = _SeqLLM(["weak [1]", "still weak [1]"])
    harness = _FakeHarness([0.3, 0.4])
    service = _harness_service(ranked=[_scored("a", "x")], llm=llm, harness=harness)

    result = service.answer(Query(text="q"))

    assert result.answered is False
    assert "cannot answer" in result.answer.lower()
    assert result.grounding_confidence == 0.4  # carries the best seen
    assert llm.calls == 2


def test_harness_drops_unsupported_citations() -> None:
    llm = _SeqLLM(["A [1] B [2]"])
    harness = _FakeHarness([0.9], drop={2})
    service = _harness_service(
        ranked=[_scored("a", "x"), _scored("b", "y")], llm=llm, harness=harness
    )

    result = service.answer(Query(text="q"))

    assert [c.id for c in result.citations] == [1]


def test_harness_not_applied_to_model_refusal() -> None:
    llm = _SeqLLM(["I cannot answer from the provided documents."])
    harness = _FakeHarness([0.9])
    service = _harness_service(ranked=[_scored("a", "x")], llm=llm, harness=harness)

    result = service.answer(Query(text="q"))

    assert result.answered is False
    assert harness.calls == 0  # a refusal short-circuits before verification
