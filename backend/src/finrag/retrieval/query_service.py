"""Top-level query service: route a question to the exact or narrative path.

Composes the pieces into one entry point. The router decides whether a question
wants an exact figure or a narrative answer:

  * **exact** -> resolve the company, read the fact from DuckDB, and format a
    deterministic cited answer (no LLM, so the number is exactly as filed). If no
    fact is found, fall through to narrative rather than failing.
  * **narrative** -> the retrieve -> rerank -> grounded-generation path.

Both paths return the same :class:`GroundedAnswer`, so the endpoint and the eval
harness treat them uniformly.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from finrag.core.interfaces.crosscutting import (
    InputGuard,
    OutputGuard,
    PiiRedactor,
    Tracer,
)
from finrag.core.interfaces.retrieval import QueryTransform, Router, StructuredQA
from finrag.core.types import Citation, FinancialFact, Query
from finrag.generation.token_stream import suppressed
from finrag.observability import NullTracer
from finrag.retrieval.answer import AnswerService, GroundedAnswer, SourceCard
from finrag.retrieval.structured_qa import resolve_metric
from finrag.retrieval.understanding import resolve_entity

_EXACT = "exact"
# Returned when the input guard blocks a query — no retrieval, no generation.
_BLOCKED = "I can't help with that request."
# Returned when the output guard blocks a produced answer before it is shown.
_BLOCKED_OUTPUT = "I'm not able to share an answer to that."


class _CollectionResolver(Protocol):
    """The slice of the structured store used to name the answering company."""

    def find_collection(self, text: str) -> tuple[str, str] | None: ...


def _format_value(value: float, unit: str) -> str:
    """Render a figure exactly, with a ``$`` for USD and the unit otherwise."""
    body = f"{int(value):,}" if value == int(value) else f"{value:,.2f}"
    if unit == "USD":
        return f"${body}"
    return f"{body} {unit}" if unit else body


def _fiscal_label(period: str) -> str:
    """Render a period as ``FY<year>`` from its end date, or the raw period."""
    try:
        return f"FY{date.fromisoformat(period.split('/')[-1]).year}"
    except ValueError:
        return period


def _format_exact(company: str, metric: str, fact: FinancialFact) -> GroundedAnswer:
    """Build a deterministic, cited answer from a single exact fact."""
    formatted = _format_value(fact.value, fact.unit)
    period_label = _fiscal_label(fact.period)
    source = SourceCard(
        id=1,
        chunk_id=fact.fact_id,
        title=f"{company} · {metric} · {period_label}",
        url=f"/filings/{fact.filing_id}",
        snippet=f"{fact.concept} = {formatted}",
    )
    return GroundedAnswer(
        answer=f"{company} — {metric}, {period_label}: {formatted} [1]",
        answered=True,
        citations=[Citation(id=1, source_doc_id=fact.filing_id, period=fact.period)],
        sources=[source],
    )


class QueryService:
    """Routes a query to the exact or narrative path and returns one answer shape."""

    def __init__(
        self,
        *,
        router: Router,
        structured_qa: StructuredQA,
        narrative: AnswerService,
        store: _CollectionResolver,
        query_transform: QueryTransform | None = None,
        tracer: Tracer | None = None,
        input_guard: InputGuard | None = None,
        output_guard: OutputGuard | None = None,
        pii_redactor: PiiRedactor | None = None,
    ) -> None:
        """Compose the query service from its router, exact QA, and narrative path.

        Args:
          router: Decides exact vs narrative.
          structured_qa: Reads exact figures from the structured store.
          narrative: The retrieve -> rerank -> generate path (and the fallback).
          store: Resolves the company named in the query, for the exact answer.
          query_transform: Optional query understanding (history-aware rewrite),
            applied before entity resolution and routing.
          tracer: Records the root query span; the narrative path's spans nest
            under it. Defaults to a no-op.
          input_guard: Optional injection/jailbreak screen; a blocked query is
            refused before any retrieval or generation runs.
          output_guard: Optional answer screen; a produced answer that leaks the
            prompt scaffold or carries unsafe content is replaced with a refusal
            before it is returned.
          pii_redactor: Optional privacy pass; detected PII in the answer and its
            source snippets is masked before the answer is returned.
        """
        self._router = router
        self._structured_qa = structured_qa
        self._narrative = narrative
        self._store = store
        self._query_transform = query_transform
        self._tracer = tracer or NullTracer()
        self._input_guard = input_guard
        self._output_guard = output_guard
        self._pii_redactor = pii_redactor

    def answer(self, query: Query) -> GroundedAnswer:
        """Answer a query via the routed path, falling back to narrative.

        The query is first understood — rewritten to stand alone (if there is
        history) and scoped to its company — then routed.

        Args:
          query: The user question and its metadata filters.

        Returns:
          The grounded answer from whichever path produced one.
        """
        with self._tracer.span("query") as root:
            blocked = self._guard(query)
            if blocked is not None:
                root.set(path="guard")
                return blocked
            with self._tracer.span("understand"):
                query = self._understand(query)
            with self._tracer.span("route") as route_span:
                routed_exact = _EXACT in self._router.route(query)
                route_span.set(path=_EXACT if routed_exact else "narrative")
            if routed_exact:
                with self._tracer.span("exact"):
                    exact = self._exact_answer(query)
                if exact is not None:
                    root.set(path=_EXACT)
                    return self._finalize_output(exact)
            root.set(path="narrative")
            return self._finalize_output(self._narrative.answer(query))

    def _guard(self, query: Query) -> GroundedAnswer | None:
        """Screen the query; return a refusal if blocked, else ``None`` to proceed."""
        if self._input_guard is None:
            return None
        with self._tracer.span("guard") as guard_span:
            verdict = self._input_guard.inspect(query.text)
            guard_span.set(allowed=verdict.allowed, category=verdict.category or "")
            if verdict.allowed:
                return None
        return GroundedAnswer(answer=_BLOCKED, answered=False)

    def _finalize_output(self, answer: GroundedAnswer) -> GroundedAnswer:
        """Apply the output-side guarantees before the answer leaves the service.

        First the safety screen (which may replace the answer with a refusal),
        then the privacy redaction over whatever survives. A blocked answer is a
        fixed refusal string, so redacting it is a harmless no-op.
        """
        return self._redact_output(self._screen_output(answer))

    def _screen_output(self, answer: GroundedAnswer) -> GroundedAnswer:
        """Screen a produced answer; replace it with a refusal if the guard blocks.

        The authoritative output check: it vets the whole answer text and, on a
        block, drops the answer along with its citations and sources so nothing
        from the unsafe draft leaks. The ``/chat`` path screens streamed segments
        with the same guard, but this is the final word on the returned answer.
        """
        if self._output_guard is None:
            return answer
        with self._tracer.span("output_guard") as guard_span:
            verdict = self._output_guard.screen(answer.answer)
            guard_span.set(allowed=verdict.allowed, category=verdict.category or "")
            if verdict.allowed:
                return answer
        return GroundedAnswer(answer=_BLOCKED_OUTPUT, answered=False)

    def _redact_output(self, answer: GroundedAnswer) -> GroundedAnswer:
        """Mask any PII in the answer text and its source snippets.

        The authoritative privacy pass over the whole answer (PII can span tokens,
        so it cannot be done reliably segment-by-segment). The ``/chat`` path
        redacts streamed segments with the same redactor for the high-confidence
        local identifiers; this is the final word on the returned answer.
        """
        if self._pii_redactor is None:
            return answer
        with self._tracer.span("pii_redact") as span:
            answer_red = self._pii_redactor.redact(answer.answer)
            snippets = [(card, self._pii_redactor.redact(card.snippet)) for card in answer.sources]
            entities = sorted(
                {*answer_red.entities, *(e for _, red in snippets for e in red.entities)}
            )
            span.set(entities=",".join(entities), count=len(entities))
            if not entities:
                return answer
            sources = [card.model_copy(update={"snippet": red.text}) for card, red in snippets]
        return answer.model_copy(
            update={"answer": answer_red.text, "sources": sources, "redacted": entities}
        )

    def _understand(self, query: Query) -> Query:
        """Rewrite the query to stand alone (if history) and scope it to its company."""
        if self._query_transform is not None:
            # A rewrite is an internal LLM call; keep its tokens out of the stream.
            with suppressed():
                query = self._query_transform.transform(query)
        return resolve_entity(query, self._store)

    def _exact_answer(self, query: Query) -> GroundedAnswer | None:
        """Try the exact path; return ``None`` to fall back to narrative."""
        found = self._store.find_collection(query.text)
        collection_id = query.filters.get("collection_id") or (found[0] if found else None)
        if collection_id is None:
            return None
        company = found[1] if found else collection_id

        scoped = Query(
            text=query.text,
            history=query.history,
            filters={**query.filters, "collection_id": collection_id},
        )
        facts = self._structured_qa.answer(scoped)
        if not facts:
            return None

        metric = resolve_metric(query.text)
        label = metric[0] if metric else facts[0].concept
        return _format_exact(company, label, facts[0])
