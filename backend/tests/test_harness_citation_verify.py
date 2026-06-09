"""Tests for the citation-verify harness step (hermetic: fake LLM judge).

Pins the hybrid contract: deterministic numeric grounding caps confidence, the
judge drops the citations it flags, confidence is the lower of the two signals,
and an unparseable verdict degrades safely.
"""

from __future__ import annotations

from finrag.core.interfaces.harness import HarnessStep
from finrag.core.registry import registry
from finrag.core.types import (
    Chunk,
    ChunkType,
    Citation,
    DocumentMetadata,
    Evidence,
    GenerationResult,
    LLMResponse,
    Market,
    ScoredChunk,
    Usage,
)
from finrag.harness.citation_verify import CitationVerifyStep


class _FakeJudge:
    """Returns a fixed verdict string, ignoring the prompt."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(text=self.reply, usage=Usage())


def _evidence(*texts: str) -> Evidence:
    chunks = []
    for i, text in enumerate(texts):
        md = DocumentMetadata(
            collection_id="c",
            company_name="Co",
            market=Market.US,
            filing_type="10-K",
            source_doc_id=f"d{i}",
        )
        chunks.append(
            ScoredChunk(
                chunk=Chunk(chunk_id=f"c{i}", text=text, chunk_type=ChunkType.TEXT, metadata=md),
                score=1.0,
            )
        )
    return Evidence(chunks=chunks)


def _draft(answer: str, cited: list[int]) -> GenerationResult:
    return GenerationResult(
        answer=answer, citations=[Citation(id=i, source_doc_id="d") for i in cited]
    )


def test_judge_drops_flagged_citations() -> None:
    step = CitationVerifyStep(_FakeJudge("CONFIDENCE: 0.9\nUNSUPPORTED: 2"))
    draft = _draft("Claim A [1] and claim B [2].", cited=[1, 2])

    out = step.apply(draft, _evidence("source one", "source two"))

    assert [c.id for c in out.citations] == [1]  # [2] dropped by the judge
    assert out.grounding_confidence == 0.9  # no figures -> numeric 1.0, min(1.0, 0.9)


def test_numeric_grounding_caps_confidence() -> None:
    step = CitationVerifyStep(_FakeJudge("CONFIDENCE: 1.0\nUNSUPPORTED: none"))
    # Answer states 999, which appears in no source -> numeric grounding 0.
    out = step.apply(_draft("Revenue was 999 [1].", cited=[1]), _evidence("Revenue was 4,250."))

    assert out.grounding_confidence == 0.0  # min(0.0 numeric, 1.0 judge)


def test_grounded_figure_passes() -> None:
    step = CitationVerifyStep(_FakeJudge("CONFIDENCE: 0.8\nUNSUPPORTED: none"))
    out = step.apply(
        _draft("Revenue was 4,250 [1].", cited=[1]), _evidence("Revenue was 4,250 million.")
    )

    assert out.grounding_confidence == 0.8  # figure grounded -> numeric 1.0, min(1.0, 0.8)
    assert [c.id for c in out.citations] == [1]


def test_unparseable_verdict_degrades_safely() -> None:
    step = CitationVerifyStep(_FakeJudge("the model rambled without the format"))
    out = step.apply(_draft("A claim [1].", cited=[1]), _evidence("some source"))

    assert out.grounding_confidence == 0.5  # default confidence
    assert [c.id for c in out.citations] == [1]  # nothing flagged -> nothing dropped


def test_registered_and_conforms_to_protocol() -> None:
    adapter = registry.create("harness", "citation_verify", llm=_FakeJudge("CONFIDENCE: 1.0"))
    assert isinstance(adapter, CitationVerifyStep)
    assert isinstance(adapter, HarnessStep)
