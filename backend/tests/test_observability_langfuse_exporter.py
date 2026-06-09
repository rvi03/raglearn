"""Tests for the Langfuse/OpenTelemetry trace exporter.

Hermetic: the OTel tracer is backed by an in-memory span exporter, so the
Span-tree → OTel-span mapping is verified end to end without a live Langfuse.
Only the final OTLP/HTTP hop (built by ``LangfuseSpanExporter.build``) needs the
container, and is not exercised here.
"""

from __future__ import annotations

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from finrag.core.types import CostBreakdown, Span, Usage
from finrag.observability.langfuse_exporter import LangfuseSpanExporter


def _exporter() -> tuple[LangfuseSpanExporter, InMemorySpanExporter]:
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    tracer = provider.get_tracer("test")
    # Fixed wall clock so timestamps are deterministic.
    return LangfuseSpanExporter(tracer, now_ns=lambda: 1_000_000_000), memory


def test_maps_tree_to_otel_spans_with_genai_attributes() -> None:
    exporter, memory = _exporter()
    root = Span(
        name="query",
        latency_ms=250.0,
        attributes={"path": "narrative"},
        children=[
            Span(
                name="generate",
                latency_ms=200.0,
                usage=Usage(tokens_in=12, tokens_out=8),
                model="qwen2.5",
                cost=CostBreakdown(model="qwen2.5", tokens_in=12, tokens_out=8, usd=0.15),
            )
        ],
    )

    exporter(root)

    spans = {s.name: s for s in memory.get_finished_spans()}
    assert set(spans) == {"query", "generate"}

    # Parent/child structure is preserved.
    assert spans["generate"].parent is not None
    assert spans["generate"].parent.span_id == spans["query"].context.span_id

    # GenAI + cost attributes land on the generate span.
    gen_attrs = spans["generate"].attributes or {}
    assert gen_attrs["gen_ai.request.model"] == "qwen2.5"
    assert gen_attrs["gen_ai.usage.input_tokens"] == 12
    assert gen_attrs["gen_ai.usage.output_tokens"] == 8
    assert gen_attrs["finrag.cost.usd"] == 0.15

    # Our own attributes are namespaced.
    assert (spans["query"].attributes or {})["finrag.path"] == "narrative"


def test_span_duration_matches_latency() -> None:
    exporter, memory = _exporter()
    exporter(Span(name="query", latency_ms=250.0))

    span = memory.get_finished_spans()[0]
    assert span.end_time - span.start_time == 250 * 1_000_000  # ns


def test_error_status_propagates() -> None:
    exporter, memory = _exporter()
    root = Span(name="query", latency_ms=10.0, children=[Span(name="retrieve", status="error")])

    exporter(root)

    spans = {s.name: s for s in memory.get_finished_spans()}
    assert spans["retrieve"].status.status_code == StatusCode.ERROR
    assert spans["query"].status.status_code != StatusCode.ERROR


def test_non_llm_span_has_no_token_attributes() -> None:
    exporter, memory = _exporter()
    exporter(Span(name="retrieve", latency_ms=5.0))  # no usage, no model

    attrs = memory.get_finished_spans()[0].attributes or {}
    assert "gen_ai.request.model" not in attrs
    assert "gen_ai.usage.input_tokens" not in attrs


def test_anchors_to_the_request_trace_id() -> None:
    from finrag.observability.tracer import reset_trace_id, set_trace_id

    exporter, memory = _exporter()
    trace_id = "0123456789abcdef0123456789abcdef"  # 32 hex = a 128-bit OTel id
    token = set_trace_id(trace_id)
    try:
        exporter(Span(name="query", latency_ms=5.0, children=[Span(name="generate")]))
    finally:
        reset_trace_id(token)

    spans = memory.get_finished_spans()
    # Every span in the tree carries the caller-assigned trace id (→ Langfuse id).
    assert {format(s.context.trace_id, "032x") for s in spans} == {trace_id}
