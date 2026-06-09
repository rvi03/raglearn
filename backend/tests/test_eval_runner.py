"""Tests for the retrieval eval runner (hermetic: fake service, embedder, store).

No models or vector store run. Fakes feed the runner controlled answers so the
scoring is pinned: context recall vs citation correctness are distinguished
(retrieving the right chunk is not the same as citing it), figures are matched,
abstentions score zero, and ingest embeds + upserts the corpus.
"""

from __future__ import annotations

from collections.abc import Sequence

from finrag.core.types import (
    Chunk,
    ChunkType,
    Citation,
    DocumentMetadata,
    EmbeddingVector,
    Market,
)
from finrag.eval.golden import GoldenCase
from finrag.eval.runner import (
    evaluate_case,
    ingest_corpus,
    ingest_facts,
    run_eval,
    summarize,
)
from finrag.eval.synthetic import synthetic_facts
from finrag.retrieval.answer import GroundedAnswer, SourceCard
from finrag.stores.duckdb_structured import DuckdbStructuredStore


def _source(source_id: int, chunk_id: str) -> SourceCard:
    return SourceCard(id=source_id, chunk_id=chunk_id, title="t", url="/u", snippet="s")


def _answer(
    text: str, sources: list[SourceCard], cited_ids: list[int], *, answered: bool = True
) -> GroundedAnswer:
    return GroundedAnswer(
        answer=text,
        answered=answered,
        citations=[Citation(id=i, source_doc_id="d") for i in cited_ids],
        sources=sources,
    )


_CASE = GoldenCase(
    id="nw-revenue",
    query="Northwind revenue?",
    expected_chunk_ids=["nw-revenue"],
    expected_values=["4,250"],
)


def test_perfect_answer_scores_one() -> None:
    answer = _answer("Revenue was $4,250 million [1].", [_source(1, "nw-revenue")], cited_ids=[1])

    result = evaluate_case(_CASE, answer)

    assert result.answered is True
    assert result.abstention_correct is True  # answerable case, and it answered
    assert result.context_recall == 1.0
    assert result.citation_correctness == 1.0
    assert result.numeric_match == 1.0


def test_wrong_source_scores_zero() -> None:
    answer = _answer("Revenue was $880 million [1].", [_source(1, "ac-revenue")], cited_ids=[1])

    result = evaluate_case(_CASE, answer)

    assert result.context_recall == 0.0  # wrong chunk retrieved
    assert result.citation_correctness == 0.0
    assert result.numeric_match == 0.0  # wrong figure


def test_retrieved_but_not_cited_splits_the_two_metrics() -> None:
    # The right chunk is in the evidence, but the answer cites nothing.
    answer = _answer("Revenue was $4,250 million.", [_source(1, "nw-revenue")], cited_ids=[])

    result = evaluate_case(_CASE, answer)

    assert result.context_recall == 1.0  # retrieved it
    assert result.citation_correctness == 0.0  # but did not cite it
    assert result.numeric_match == 1.0


def test_wrongly_abstaining_on_answerable_is_penalized() -> None:
    # An answerable question the system refused: abstention is wrong, and with no
    # sources kept there is nothing recalled or cited.
    answer = _answer("I cannot answer.", sources=[], cited_ids=[], answered=False)

    result = evaluate_case(_CASE, answer)  # _CASE is answerable

    assert result.answered is False
    assert result.abstention_correct is False  # should have answered
    assert result.context_recall == 0.0
    assert result.citation_correctness == 0.0
    assert result.numeric_match == 0.0


def test_correct_abstention_on_unanswerable_scores_right() -> None:
    case = GoldenCase(id="neg", query="who is the CEO?", answerable=False)
    answer = _answer("I cannot answer.", sources=[], cited_ids=[], answered=False)

    result = evaluate_case(case, answer)

    assert result.abstention_correct is True  # correctly abstained


def test_hallucinating_on_unanswerable_is_caught() -> None:
    case = GoldenCase(id="neg", query="who is the CEO?", answerable=False)
    answer = _answer("The CEO is Jane Doe [1].", [_source(1, "x")], cited_ids=[1])

    result = evaluate_case(case, answer)

    assert result.abstention_correct is False  # answered when it should have abstained


class _FakeService:
    """Returns a pre-set answer per query text."""

    def __init__(self, answers: dict[str, GroundedAnswer]) -> None:
        self._answers = answers

    def answer(self, query: object) -> GroundedAnswer:
        return self._answers[query.text]  # type: ignore[attr-defined]


def test_run_eval_and_summarize_average_across_cases() -> None:
    cases = [
        GoldenCase(id="a", query="qa", expected_chunk_ids=["a"], expected_values=["1"]),
        GoldenCase(id="b", query="qb", expected_chunk_ids=["b"], expected_values=["2"]),
        GoldenCase(id="c", query="qc", answerable=False),  # correctly abstains
        GoldenCase(id="d", query="qd", answerable=False),  # hallucinates instead
    ]
    service = _FakeService(
        {
            "qa": _answer("value is 1 [1]", [_source(1, "a")], cited_ids=[1]),  # perfect
            "qb": _answer("value is 9 [1]", [_source(1, "wrong")], cited_ids=[1]),  # all wrong
            "qc": _answer("I cannot answer.", sources=[], cited_ids=[], answered=False),  # right
            "qd": _answer("made up [1]", [_source(1, "x")], cited_ids=[1]),  # wrong: answered
        }
    )

    results = run_eval(cases, service)  # type: ignore[arg-type]
    summary = summarize(results)

    assert [r.case_id for r in results] == ["a", "b", "c", "d"]
    # a,b,c correct on abstention; d hallucinated -> 3/4
    assert summary["abstention_accuracy"] == 0.75
    # quality metrics averaged over answerable cases (a, b) only
    assert summary["context_recall"] == 0.5  # 1.0 + 0.0
    assert summary["citation_correctness"] == 0.5
    assert summary["numeric_match"] == 0.5


def test_summarize_empty_is_empty() -> None:
    assert summarize([]) == {}


def test_summarize_averages_grounding_when_present() -> None:
    cases = [
        GoldenCase(id="a", query="qa", expected_chunk_ids=["a"]),
        GoldenCase(id="b", query="qb", expected_chunk_ids=["b"]),
    ]
    grounded = GroundedAnswer(
        answer="x [1]", sources=[_source(1, "a")], citations=[Citation(id=1, source_doc_id="d")]
    )
    grounded.grounding_confidence = 0.8
    plain = _answer("y [1]", [_source(1, "b")], cited_ids=[1])  # no grounding_confidence

    results = run_eval(cases, _FakeService({"qa": grounded, "qb": plain}))  # type: ignore[arg-type]
    summary = summarize(results)

    assert summary["grounding"] == 0.8  # averaged over the verified case only


class _FakeEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        self.texts = list(texts)
        return [EmbeddingVector(dense=[1.0]) for _ in texts]


class _FakeStore:
    def __init__(self) -> None:
        self.upserted: list[Chunk] = []

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[EmbeddingVector]) -> None:
        assert len(chunks) == len(vectors)
        self.upserted = list(chunks)

    def search(
        self,
        vector: EmbeddingVector,
        *,
        top_k: int,
        filters: dict[str, str],
        access_tags: Sequence[str] = (),
    ) -> list[object]:
        raise AssertionError("ingest must not search")


def test_ingest_corpus_embeds_and_upserts() -> None:
    md = DocumentMetadata(
        collection_id="c",
        company_name="Co",
        market=Market.US,
        filing_type="Annual Report",
        source_doc_id="synthetic/x",
    )
    chunks = [Chunk(chunk_id="a", text="alpha", chunk_type=ChunkType.TEXT, metadata=md)]
    embedder = _FakeEmbedder()
    store = _FakeStore()

    ingest_corpus(chunks, embedder, store)

    assert embedder.texts == ["alpha"]  # chunk text embedded
    assert [c.chunk_id for c in store.upserted] == ["a"]


def test_ingest_facts_writes_to_structured_store() -> None:
    store = DuckdbStructuredStore(":memory:")
    try:
        ingest_facts(synthetic_facts(), store)
        facts = store.query_facts("us-cik-901", ["us-gaap:Revenues"])
        assert any(f.fact_id == "nw-fact-rev-2024" for f in facts)
    finally:
        store.close()
