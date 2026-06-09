"""Minimal grounded answering: retrieve → rerank → generate a cited answer.

The narrative slice's end-to-end orchestrator. It is *minimal* by design — the
full answer-quality harness (citation/groundedness verification, self-consistency,
reflection, judge-gate) is the Generation vertical's job. Here we:

  1. retrieve a candidate pool (hybrid vector search),
  2. rerank it with a cross-encoder and keep the top few,
  3. stuff those into an evidence-only prompt and ask the LLM to answer with
     ``[n]`` citations,
  4. map the ``[n]`` markers back to source provenance.

Grounding is enforced in the prompt: instruction and data are separated, the
model is told to use *only* the numbered sources and to abstain when they do not
contain the answer. If retrieval finds nothing, we abstain without calling the
LLM at all.

This is composed from the resolved stage adapters (``retriever``, ``reranker``,
``llm_backend``) in the API layer rather than registered as its own adapter — it
is wiring over stages, not a stage with config alternatives.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from finrag.core.interfaces.crosscutting import Tracer
from finrag.core.interfaces.generation import LLMBackend
from finrag.core.interfaces.harness import HarnessStep
from finrag.core.interfaces.retrieval import Reranker, Retriever
from finrag.core.types import (
    Citation,
    DocumentMetadata,
    Evidence,
    GenerationResult,
    Query,
    ScoredChunk,
    Usage,
)
from finrag.generation.token_stream import suppressed
from finrag.observability import NullTracer

# Candidates pulled by retrieval before reranking, and kept after — a cross-encoder
# is precise but costly per pair, so retrieve broad, rerank, then keep a tight set.
_DEFAULT_CANDIDATE_K = 20
_DEFAULT_TOP_K = 5
# Characters of a chunk shown as its evidence-card snippet.
_SNIPPET_CHARS = 240
# Shown when retrieval returns nothing — we never fabricate an ungrounded answer.
_ABSTAIN = "I cannot answer this question from the available documents."
# Matches an inline ``[n]`` citation marker in the model's answer.
_CITE_RE = re.compile(r"\[(\d+)\]")
# Default verify→regenerate budget and the grounding bar an answer must clear.
_DEFAULT_MAX_ATTEMPTS = 2
_DEFAULT_GROUNDING_THRESHOLD = 0.6
# Appended to the prompt on a retry, to push the model to stay on-evidence.
_REGEN_REINFORCE = (
    "\n\nYour previous answer was not fully supported by the sources. Answer again "
    "using ONLY the numbered sources, state figures exactly as written, and cite "
    "every claim."
)


class SourceCard(BaseModel):
    """One piece of evidence shown beside an answer (the ``[n]`` the model cites)."""

    id: int  # the source number used in the prompt and in ``[n]`` markers
    chunk_id: str  # the underlying chunk's stable id (provenance; used by eval)
    title: str
    url: str
    snippet: str


class GroundedAnswer(BaseModel):
    """A cited answer plus the evidence it was grounded in and its token usage."""

    answer: str
    answered: bool = True  # False = abstained (no evidence, or the model judged it insufficient)
    citations: list[Citation] = Field(default_factory=list)
    sources: list[SourceCard] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    grounding_confidence: float | None = None  # set when a harness verified the answer
    redacted: list[str] = Field(default_factory=list)  # PII entity types masked in the answer


def _source_title(metadata: DocumentMetadata) -> str:
    """Build a human label for a source from its provenance fields."""
    period = metadata.fiscal_period or (
        str(metadata.fiscal_year) if metadata.fiscal_year is not None else None
    )
    parts = [metadata.company_name, metadata.filing_type, period, metadata.section]
    return " · ".join(part for part in parts if part)


def _source_url(metadata: DocumentMetadata) -> str:
    """Build the source-viewer URL, anchored to the page when known."""
    url = f"/sources/{metadata.source_doc_id}"
    return f"{url}#p{metadata.page}" if metadata.page is not None else url


def _snippet(text: str) -> str:
    """Return a short, single-line preview of a chunk's text."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _SNIPPET_CHARS:
        return collapsed
    return collapsed[:_SNIPPET_CHARS].rstrip() + "…"


def _source_card(source_id: int, scored: ScoredChunk) -> SourceCard:
    """Render one ranked chunk as a numbered evidence card."""
    metadata = scored.chunk.metadata
    return SourceCard(
        id=source_id,
        chunk_id=scored.chunk.chunk_id,
        title=_source_title(metadata),
        url=_source_url(metadata),
        snippet=_snippet(scored.chunk.text),
    )


def _build_prompt(query: Query, ranked: Sequence[ScoredChunk], feedback: str = "") -> str:
    """Assemble the evidence-only grounding prompt with numbered sources.

    Instruction and data are kept in separate, labelled blocks so the model
    cannot mistake document text for instructions, and the sources are numbered
    so the model can cite them as ``[n]``. ``feedback`` is appended on a retry to
    steer the model back onto the evidence.
    """
    sources = "\n\n".join(
        f"[{i}] ({_source_title(scored.chunk.metadata)})\n{scored.chunk.text}"
        for i, scored in enumerate(ranked, start=1)
    )
    return (
        "You are a financial-analysis assistant. Answer the QUESTION using ONLY "
        "the numbered SOURCES below. Cite every claim with its source number in "
        "square brackets, e.g. [1]. If the sources do not contain the answer, "
        f'reply with exactly "{_ABSTAIN}" and cite nothing. Do not use outside '
        "knowledge.\n\n"
        f"SOURCES:\n{sources}\n\n"
        f"QUESTION: {query.text}{feedback}\n\n"
        "ANSWER:"
    )


def _looks_like_abstention(answer_text: str) -> bool:
    """Return whether the model refused to answer from the evidence.

    Tolerant of the model appending stray ``[n]`` markers or rephrasing slightly:
    the signal is the "cannot answer" core of the instructed abstention sentence.
    """
    normalized = _CITE_RE.sub("", answer_text).strip().lower()
    return "cannot answer" in normalized


def _citations(answer_text: str, ranked: Sequence[ScoredChunk]) -> list[Citation]:
    """Map the ``[n]`` markers the model emitted back to source provenance.

    Only sources the answer actually cites become citations, and only ids that
    refer to a real source are kept (a hallucinated ``[9]`` over five sources is
    dropped). Order follows the source numbering for stable output.
    """
    cited_ids = {int(marker) for marker in _CITE_RE.findall(answer_text)}
    citations: list[Citation] = []
    for source_id, scored in enumerate(ranked, start=1):
        if source_id not in cited_ids:
            continue
        metadata = scored.chunk.metadata
        citations.append(
            Citation(
                id=source_id,
                source_doc_id=metadata.source_doc_id,
                page=metadata.page,
                section=metadata.section,
                period=metadata.fiscal_period,
            )
        )
    return citations


class AnswerService:
    """Orchestrates retrieve → rerank → grounded generation into a cited answer."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        reranker: Reranker,
        llm: LLMBackend,
        harness: HarnessStep | None = None,
        tracer: Tracer | None = None,
        candidate_k: int = _DEFAULT_CANDIDATE_K,
        top_k: int = _DEFAULT_TOP_K,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        grounding_threshold: float = _DEFAULT_GROUNDING_THRESHOLD,
    ) -> None:
        """Compose the narrative answer path from its stage adapters.

        Args:
          retriever: Produces the candidate pool (hybrid vector search).
          reranker: Re-scores the pool; the top ``top_k`` are kept as evidence.
          llm: Generates the grounded answer from the evidence.
          harness: Optional answer-quality step; when set, the answer is verified
            and regenerated until it clears ``grounding_threshold``.
          tracer: Records spans over retrieve/rerank/generate/harness. Defaults to
            a no-op so callers that do not observe stay unchanged.
          candidate_k: How many candidates to retrieve before reranking.
          top_k: How many reranked chunks to ground the answer in.
          max_attempts: Generate→verify→regenerate budget (only with a harness).
          grounding_threshold: Minimum grounding confidence to accept an answer.
        """
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm
        self._harness = harness
        self._tracer = tracer or NullTracer()
        self._candidate_k = candidate_k
        self._top_k = top_k
        self._max_attempts = max_attempts
        self._grounding_threshold = grounding_threshold

    def answer(self, query: Query) -> GroundedAnswer:
        """Answer a query from the indexed corpus, with citations and sources.

        Returns an abstention (no LLM call) when retrieval finds no evidence. With
        a harness configured, the draft is verified and regenerated until it
        clears the grounding threshold, else the answer abstains.

        Args:
          query: The user question and its metadata filters.

        Returns:
          The grounded answer, its citations, the evidence sources, and usage.
        """
        with self._tracer.span("narrative"):
            with self._tracer.span("retrieve", candidate_k=self._candidate_k):
                candidates = self._retriever.retrieve(query, top_k=self._candidate_k)
            with self._tracer.span("rerank", top_k=self._top_k) as rerank_span:
                ranked = self._reranker.rerank(query, candidates)[: self._top_k]
                rerank_span.set(kept=len(ranked))
            if not ranked:
                # Nothing retrieved — abstain without spending an LLM call.
                return GroundedAnswer(answer=_ABSTAIN, answered=False)

            sources = [_source_card(i, scored) for i, scored in enumerate(ranked, start=1)]
            if self._harness is None:
                return self._single_shot(query, ranked, sources)
            return self._verified(query, ranked, sources)

    def _single_shot(
        self, query: Query, ranked: Sequence[ScoredChunk], sources: list[SourceCard]
    ) -> GroundedAnswer:
        """Generate once, with no answer-quality verification (harness disabled)."""
        with self._tracer.span("generate") as gen:
            response = self._llm.generate(_build_prompt(query, ranked))
            gen.record_usage(response.usage, response.model)
        if _looks_like_abstention(response.text):
            return GroundedAnswer(
                answer=_ABSTAIN, answered=False, sources=sources, usage=response.usage
            )
        return GroundedAnswer(
            answer=response.text,
            answered=True,
            citations=_citations(response.text, ranked),
            sources=sources,
            usage=response.usage,
        )

    def _verified(
        self, query: Query, ranked: Sequence[ScoredChunk], sources: list[SourceCard]
    ) -> GroundedAnswer:
        """Generate → verify → regenerate until grounded, else abstain.

        Each attempt drafts an answer, the harness scores its grounding and drops
        unsupported citations, and the first draft to clear the threshold is
        returned. A model refusal short-circuits to an abstention; exhausting the
        attempt budget abstains too, carrying the best confidence seen.
        """
        assert self._harness is not None
        evidence = Evidence(chunks=list(ranked))
        feedback = ""
        best: GenerationResult | None = None
        for _attempt in range(self._max_attempts):
            with self._tracer.span("generate") as gen:
                response = self._llm.generate(_build_prompt(query, ranked, feedback))
                gen.record_usage(response.usage, response.model)
            if _looks_like_abstention(response.text):
                return GroundedAnswer(
                    answer=_ABSTAIN, answered=False, sources=sources, usage=response.usage
                )
            draft = GenerationResult(
                answer=response.text,
                citations=_citations(response.text, ranked),
                usage=response.usage,
            )
            with self._tracer.span("harness"), suppressed():
                # The judge is an internal LLM call; its tokens must not leak into
                # the user-facing answer stream.
                verified = self._harness.apply(draft, evidence)
            confidence = verified.grounding_confidence
            if best is None or (confidence or 0.0) > (best.grounding_confidence or 0.0):
                best = verified
            if confidence is not None and confidence >= self._grounding_threshold:
                return GroundedAnswer(
                    answer=verified.answer,
                    answered=True,
                    citations=verified.citations,
                    sources=sources,
                    usage=verified.usage,
                    grounding_confidence=confidence,
                )
            feedback = _REGEN_REINFORCE
        # Never cleared the bar — abstain rather than return a weakly-grounded answer.
        return GroundedAnswer(
            answer=_ABSTAIN,
            answered=False,
            sources=sources,
            usage=best.usage if best else Usage(),
            grounding_confidence=best.grounding_confidence if best else None,
        )
