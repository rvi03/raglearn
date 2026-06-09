"""Cross-cutting interfaces.

Concerns that ride alongside every stage rather than sitting in the pipeline,
such as cost accounting.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from finrag.core.types import CostBreakdown, GuardVerdict, Redaction, Span, Usage


@runtime_checkable
class CostModel(Protocol):
    """Prices token usage for a model.

    Implementations price from a serverless sheet, a self-hosted ``$/hr``
    amortization, or a flat zero for local dev.
    """

    def price(self, usage: Usage, model: str) -> CostBreakdown:
        """Return the cost breakdown for a unit of usage on a model."""
        ...


@runtime_checkable
class SpanHandle(Protocol):
    """A live span the caller annotates while a stage runs."""

    def record_usage(self, usage: Usage, model: str) -> None:
        """Attribute an LLM call's token usage (and its model) to this span."""
        ...

    def set(self, **attributes: str | int | float | bool) -> None:
        """Attach descriptive attributes to this span (e.g. ``top_k=8``)."""
        ...


@runtime_checkable
class InputGuard(Protocol):
    """Screens an incoming query before it reaches the pipeline.

    Defends against prompt injection / jailbreak attempts (OWASP LLM01). A
    blocking verdict short-circuits the request with a safe refusal — no
    retrieval, no generation. Implementations range from pattern heuristics to
    a guard model (Llama Guard / NeMo).
    """

    def inspect(self, text: str) -> GuardVerdict:
        """Return whether ``text`` is safe to answer, and why not if blocked."""
        ...


@runtime_checkable
class OutputGuard(Protocol):
    """Screens a generated answer before it reaches the user.

    The output-side counterpart of :class:`InputGuard` (OWASP LLM02/LLM06): it
    catches an answer that leaks the system prompt, echoes its instructions, or
    carries unsafe content — the kinds of failure an injection induces *in the
    output* rather than the input. A blocking verdict replaces the answer with a
    safe refusal. Implementations range from pattern heuristics to a guard model.

    Screening is per-text and stateless so the same guard can vet either the whole
    answer (the ``/query`` path) or one streamed segment at a time (``/chat``).
    """

    def screen(self, text: str) -> GuardVerdict:
        """Return whether ``text`` is safe to show, and why not if blocked."""
        ...


@runtime_checkable
class PiiRedactor(Protocol):
    """Masks personally identifiable information in text before it is shown.

    The privacy counterpart of the guards (OWASP LLM06): rather than block, it
    *redacts* — leaving the answer intact but replacing detected identifiers with
    a marker. Coverage is region-aware (India and US/global identifiers), since
    the corpus spans both. Implementations range from pattern + checksum matching
    to an NER-backed engine (Presidio) behind this same interface.
    """

    def redact(self, text: str) -> Redaction:
        """Return ``text`` with any detected PII masked, and which types were found."""
        ...


@runtime_checkable
class MonitorEmitter(Protocol):
    """Publishes ingestion-pipeline progress for the live monitor view.

    The ingestion counterpart of the query trace: where the tracer streams a
    query's stage spans, this streams an upload's per-document pipeline progress —
    an ``upload`` when a batch is accepted, a ``node`` as each stage starts and
    finishes per document, and a ``doc_done`` for the document's terminal outcome.
    A subscriber relays these to the monitor DAG. The default implementation is a
    no-op, so the pipeline runs unobserved (and tests stay hermetic) unless a real
    emitter (Redis pub/sub) is wired in.
    """

    def upload(
        self, *, upload_id: str, country: str, created: str, docs: Sequence[tuple[str, str]]
    ) -> None:
        """Announce an accepted upload and the documents (``doc_id``, ``filename``) in it."""
        ...

    def node(
        self,
        *,
        upload_id: str,
        doc_id: str,
        stage: str,
        label: str,
        status: str,
        detail: str | None = None,
    ) -> None:
        """Report a pipeline stage transition for one document."""
        ...

    def doc_done(self, *, upload_id: str, doc_id: str, outcome: str) -> None:
        """Report a document's terminal outcome (indexed/facts-written/deferred/…)."""
        ...


@runtime_checkable
class SpanListener(Protocol):
    """Observes spans as they open and close, for live streaming.

    Distinct from a trace sink (which receives the finished tree at the end): a
    listener fires per span, so a consumer — the ``/chat`` SSE stream — can emit a
    step as ``running`` when it opens and ``done``/``failed`` when it closes.
    """

    def on_open(self, name: str, attributes: dict[str, str | int | float | bool]) -> None:
        """Called when a span named ``name`` opens."""
        ...

    def on_close(self, span: Span) -> None:
        """Called when a span closes, with the finished span."""
        ...


@runtime_checkable
class Tracer(Protocol):
    """Records nested, timed spans over a unit of work.

    One ``span`` call wraps a stage; nesting follows the ``with`` blocks, so the
    tracer reconstructs the retrieve/rerank/generate/harness tree under a root
    query span. The same tree feeds Langfuse and the UI DAG through exporters.
    """

    def span(
        self, name: str, **attributes: str | int | float | bool
    ) -> AbstractContextManager[SpanHandle]:
        """Open a span named ``name``, nested under any enclosing span."""
        ...
