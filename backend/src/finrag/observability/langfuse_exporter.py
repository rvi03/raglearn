"""Langfuse trace exporter via OpenTelemetry.

Maps a finished trace (our :class:`~finrag.core.types.Span` tree) onto
OpenTelemetry spans carrying GenAI semantic-convention attributes, and ships them
over OTLP/HTTP to Langfuse's OpenTelemetry endpoint. Langfuse renders
the retrieve/rerank/generate/harness tree with per-call tokens and cost.

Timing: our spans record an exact *duration* (monotonic) but not an absolute
wall-clock start. The exporter reconstructs OTLP timestamps from durations —
each subtree is anchored at the export instant and children are laid out
sequentially within their parent. Durations and parent/child structure are
exact; absolute alignment is approximate, which is all Langfuse's waterfall
needs.

The mapping is testable without a live Langfuse: inject any OpenTelemetry
``Tracer`` (e.g. one backed by an in-memory exporter). Only the final OTLP hop
needs the Langfuse container (the carried infra item).
"""

from __future__ import annotations

import base64
import os
import time
from collections.abc import Callable

from opentelemetry.context import Context
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
    Tracer,
    set_span_in_context,
)
from opentelemetry.trace import Span as OtelSpan

from finrag.core.errors import ConfigError
from finrag.core.types import Span
from finrag.observability.tracer import current_trace_id

# Default Langfuse OTLP path appended to the host when building from settings.
_LANGFUSE_OTLP_PATH = "/api/public/otel/v1/traces"


class LangfuseSpanExporter:
    """A trace sink that emits our span tree as OpenTelemetry spans."""

    def __init__(self, tracer: Tracer, *, now_ns: Callable[[], int] = time.time_ns) -> None:
        """Bind the exporter to an OpenTelemetry tracer.

        Args:
          tracer: The OTel tracer that records and exports spans. In production
            this is backed by an OTLP/HTTP exporter to Langfuse; tests inject one
            backed by an in-memory exporter.
          now_ns: Wall-clock nanoseconds source; injectable for deterministic
            tests.
        """
        self._tracer = tracer
        self._now_ns = now_ns

    @classmethod
    def build(cls, *, endpoint: str, headers: dict[str, str]) -> LangfuseSpanExporter:
        """Construct an exporter wired to ship over OTLP/HTTP to ``endpoint``.

        Args:
          endpoint: Full Langfuse OTLP traces URL.
          headers: OTLP headers (Langfuse uses HTTP Basic auth).
        """
        # Imported lazily so the SDK/exporter cost is only paid when Langfuse is
        # actually enabled, not on every import of this module.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
        )
        return cls(provider.get_tracer("finrag"))

    def __call__(self, root: Span) -> None:
        """Emit the finished trace as a tree of OpenTelemetry spans."""
        duration_ns = int(root.latency_ms * 1_000_000)
        # Anchor the whole tree so the root ends "now".
        start_ns = self._now_ns() - duration_ns
        self._emit(root, self._root_context(), start_ns)

    @staticmethod
    def _root_context() -> Context:
        """Parent context anchoring the trace to the request's id, when set.

        With a caller-assigned 32-hex trace id, the OTel spans inherit it, so the
        Langfuse trace id equals the id the client got on ``done`` — making the
        ``done.trace_id`` a working deep link. Without one, OTel assigns its own.
        """
        trace_id = current_trace_id()
        if not trace_id:
            return Context()
        tid = int(trace_id, 16)
        # The synthetic parent needs a non-zero span id; derive it from the id.
        span_id = int(trace_id[:16], 16) or 1
        ctx = SpanContext(
            trace_id=tid,
            span_id=span_id,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        return set_span_in_context(NonRecordingSpan(ctx))

    def _emit(self, node: Span, parent_context: Context, start_ns: int) -> None:
        """Emit one node and, sequentially within it, its children."""
        end_ns = start_ns + int(node.latency_ms * 1_000_000)
        span = self._tracer.start_span(node.name, context=parent_context, start_time=start_ns)
        self._annotate(span, node)
        child_context = set_span_in_context(span, parent_context)
        child_start = start_ns
        for child in node.children:
            self._emit(child, child_context, child_start)
            child_start += int(child.latency_ms * 1_000_000)
        span.end(end_time=end_ns)

    @staticmethod
    def _annotate(span: OtelSpan, node: Span) -> None:
        """Copy a node's usage, cost, status, and attributes onto an OTel span."""
        for key, value in node.attributes.items():
            span.set_attribute(f"finrag.{key}", value)
        if node.model:
            span.set_attribute("gen_ai.request.model", node.model)
        if node.usage.tokens_in or node.usage.tokens_out:
            span.set_attribute("gen_ai.usage.input_tokens", node.usage.tokens_in)
            span.set_attribute("gen_ai.usage.output_tokens", node.usage.tokens_out)
        if node.cost is not None:
            span.set_attribute("finrag.cost.usd", node.cost.usd)
        if node.status == "error":
            span.set_status(Status(StatusCode.ERROR))


def langfuse_headers() -> dict[str, str]:
    """Build Langfuse OTLP auth headers from the environment.

    Langfuse authenticates OTLP ingestion with HTTP Basic auth over the project
    public/secret key pair.

    Raises:
      ConfigError: If either key is missing.
    """
    public = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret = os.getenv("LANGFUSE_SECRET_KEY")
    if not public or not secret:
        raise ConfigError("langfuse exporter needs LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    token = base64.b64encode(f"{public}:{secret}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def langfuse_otlp_endpoint(host: str) -> str:
    """Return the Langfuse OTLP traces URL for a host base."""
    return host.rstrip("/") + _LANGFUSE_OTLP_PATH
