"""Citation verification: a hybrid groundedness check over a draft answer.

Two signals are combined, conservatively (the score is the lower of the two):

  * **deterministic numeric grounding** — every figure stated in the answer must
    appear verbatim in the cited evidence. Cheap, exact, and catches the classic
    hallucinated number that an LLM judge can gloss over.
  * **LLM judge** — one call that reads the numbered sources and the answer and
    returns a confidence plus the source numbers whose claim is unsupported.
    Catches paraphrased / invented prose that overlaps no figure.

The step sets ``grounding_confidence`` on the draft and drops the citations the
judge flagged as unsupported. It does *not* regenerate — the generate → verify →
regenerate loop lives in the answer service, which owns the prompt and the LLM.
"""

from __future__ import annotations

import re

from finrag.core.interfaces.generation import LLMBackend
from finrag.core.registry import registry
from finrag.core.types import Evidence, GenerationResult

# A number as written in prose: digits with optional thousands commas / decimals.
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Inline ``[n]`` citation markers — stripped before figure extraction so the
# marker's digit is not mistaken for a stated figure.
_MARKER_RE = re.compile(r"\[\d+\]")
# Judge reply lines.
_CONFIDENCE_RE = re.compile(r"confidence:\s*([01](?:\.\d+)?)", re.IGNORECASE)
_UNSUPPORTED_RE = re.compile(r"unsupported:\s*(.*)", re.IGNORECASE)
# Conservative default when the judge reply cannot be parsed.
_DEFAULT_CONFIDENCE = 0.5


def _numeric_grounding(answer: str, evidence_text: str) -> float:
    """Fraction of figures in the answer that appear verbatim in the evidence.

    Returns 1.0 when the answer states no figures (nothing to ground).
    """
    numbers = _NUMBER_RE.findall(_MARKER_RE.sub(" ", answer))
    if not numbers:
        return 1.0
    grounded = sum(1 for number in numbers if number in evidence_text)
    return grounded / len(numbers)


def _judge_prompt(answer: str, evidence: Evidence) -> str:
    """Build the fact-checking prompt: numbered sources + the answer + reply format."""
    sources = "\n\n".join(
        f"[{i}] {scored.chunk.text}" for i, scored in enumerate(evidence.chunks, start=1)
    )
    return (
        "You are a strict fact-checker. Decide whether every claim in the ANSWER is "
        "supported by the numbered SOURCES.\n\n"
        f"SOURCES:\n{sources}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Reply with exactly two lines:\n"
        "CONFIDENCE: <a number from 0 to 1>\n"
        "UNSUPPORTED: <comma-separated source numbers whose claim is not supported, or none>"
    )


def _parse_verdict(reply: str) -> tuple[float, set[int]]:
    """Parse the judge reply into ``(confidence, unsupported source ids)``."""
    confidence_match = _CONFIDENCE_RE.search(reply)
    confidence = float(confidence_match.group(1)) if confidence_match else _DEFAULT_CONFIDENCE
    unsupported: set[int] = set()
    unsupported_match = _UNSUPPORTED_RE.search(reply)
    if unsupported_match:
        unsupported = {int(n) for n in re.findall(r"\d+", unsupported_match.group(1))}
    return confidence, unsupported


@registry.register("harness", "citation_verify")
class CitationVerifyStep:
    """A :class:`~finrag.core.interfaces.HarnessStep` verifying answer grounding."""

    def __init__(self, llm: LLMBackend) -> None:
        """Bind the step to the LLM it uses as the fact-checking judge."""
        self._llm = llm

    def apply(self, draft: GenerationResult, evidence: Evidence) -> GenerationResult:
        """Score the draft's grounding and drop its unsupported citations.

        Args:
          draft: The drafted answer with its citations.
          evidence: The numbered sources the answer was generated from.

        Returns:
          The draft with ``grounding_confidence`` set (min of the deterministic
          and judge scores) and judge-flagged citations removed.
        """
        evidence_text = " ".join(scored.chunk.text for scored in evidence.chunks)
        numeric_score = _numeric_grounding(draft.answer, evidence_text)

        verdict = self._llm.generate(_judge_prompt(draft.answer, evidence))
        judge_confidence, unsupported = _parse_verdict(verdict.text)

        confidence = min(numeric_score, judge_confidence)
        kept = [citation for citation in draft.citations if citation.id not in unsupported]
        return draft.model_copy(update={"citations": kept, "grounding_confidence": confidence})
