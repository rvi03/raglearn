"""In-process tracer: builds a timed span tree for one query.

This is the default :class:`~finrag.core.interfaces.Tracer`. It needs no
external service — it times each ``span`` block, attributes LLM token usage to
the span that made the call, prices that usage through a
:class:`~finrag.core.interfaces.CostModel` when the trace closes, and hands the
finished root span to a sink. Langfuse and the Redis/SSE DAG attach later as
sinks; nothing about the call sites changes when they do.

Nesting follows the ``with`` blocks. The current span stack lives in a
:class:`~contextvars.ContextVar`, so concurrent queries (each its own async task)
build independent trees without stepping on each other.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token

from finrag.core.interfaces.crosscutting import CostModel, SpanListener
from finrag.core.logging import get_logger
from finrag.core.registry import registry
from finrag.core.types import CostBreakdown, Span, Usage

logger = get_logger(__name__)

# The chain of spans currently open on this task, outermost first. A fresh list
# is set when the root span opens and reset when it closes, so nothing leaks
# between queries.
_stack: ContextVar[list[_SpanBuilder] | None] = ContextVar("finrag_span_stack", default=None)

# An optional per-request listener notified as each span opens and closes, so a
# consumer (the /chat SSE stream) can emit live step events. Contextvar-scoped,
# so it isolates concurrent requests; propagated into worker threads by
# ``asyncio.to_thread`` (which copies the context).
_listener: ContextVar[SpanListener | None] = ContextVar("finrag_span_listener", default=None)


def set_span_listener(listener: SpanListener) -> Token[SpanListener | None]:
    """Install a per-request span listener; returns a token to reset it."""
    return _listener.set(listener)


def reset_span_listener(token: Token[SpanListener | None]) -> None:
    """Remove a previously installed span listener."""
    _listener.reset(token)


# The caller-assigned trace id (32 hex chars) for the current request, so the
# Langfuse exporter can anchor its OTel trace to a known id the client also gets
# back (on the ``done`` event) for a deep link. Contextvar-scoped; thread-copied.
_trace_id: ContextVar[str | None] = ContextVar("finrag_trace_id", default=None)


def set_trace_id(trace_id: str) -> Token[str | None]:
    """Set the current request's trace id; returns a token to reset it."""
    return _trace_id.set(trace_id)


def reset_trace_id(token: Token[str | None]) -> None:
    """Clear the current request's trace id."""
    _trace_id.reset(token)


def current_trace_id() -> str | None:
    """Return the current request's trace id, if one was set."""
    return _trace_id.get()


class _SpanBuilder:
    """A span being filled in while its ``with`` block runs.

    Implements :class:`~finrag.core.interfaces.SpanHandle`. Mutable until the
    block exits, at which point :meth:`build` freezes it into a
    :class:`~finrag.core.types.Span`.
    """

    def __init__(self, name: str, attributes: dict[str, str | int | float | bool]) -> None:
        self.name = name
        self.attributes = attributes
        self.status = "ok"
        self.latency_ms = 0.0
        self.usage = Usage()
        self.model: str | None = None
        self.children: list[Span] = []

    def record_usage(self, usage: Usage, model: str) -> None:
        """Add an LLM call's tokens to this span (summing across calls)."""
        self.usage = Usage(
            tokens_in=self.usage.tokens_in + usage.tokens_in,
            tokens_out=self.usage.tokens_out + usage.tokens_out,
        )
        self.model = model

    def set(self, **attributes: str | int | float | bool) -> None:
        """Attach descriptive attributes to this span."""
        self.attributes.update(attributes)

    def build(self, cost_model: CostModel) -> Span:
        """Freeze into a :class:`Span`, pricing usage if a model was recorded."""
        cost: CostBreakdown | None = None
        if self.model is not None:
            cost = cost_model.price(self.usage, self.model)
        return Span(
            name=self.name,
            status=self.status,
            latency_ms=self.latency_ms,
            usage=self.usage,
            model=self.model,
            cost=cost,
            attributes=self.attributes,
            children=self.children,
        )


@registry.register("tracer", "local")
class InProcessTracer:
    """A :class:`~finrag.core.interfaces.Tracer` that builds the tree in memory."""

    def __init__(
        self,
        *,
        cost_model: CostModel,
        clock: Callable[[], float] = time.perf_counter,
        sinks: Sequence[Callable[[Span], None]] = (),
    ) -> None:
        """Compose the tracer.

        Args:
          cost_model: Prices each span's token usage at trace close.
          clock: Monotonic seconds source; injectable so tests are deterministic.
          sinks: Exporters that receive the finished root span (Redis/SSE,
            Langfuse). Every trace is always logged; these run in addition. A
            sink that raises is logged and swallowed — observability never breaks
            the request it observes.
        """
        self._cost_model = cost_model
        self._clock = clock
        self._sinks = tuple(sinks)
        # Last finished root span, kept for tests and out-of-band inspection.
        self.last_trace: Span | None = None

    @contextmanager
    def span(self, name: str, **attributes: str | int | float | bool) -> Iterator[_SpanBuilder]:
        """Open a span named ``name``, nested under any enclosing span."""
        builder = _SpanBuilder(name, dict(attributes))
        stack = _stack.get()
        token = None
        if stack is None:
            stack = []
            token = _stack.set(stack)
        parent = stack[-1] if stack else None
        stack.append(builder)

        listener = _listener.get()
        if listener is not None:
            listener.on_open(name, builder.attributes)

        start = self._clock()
        try:
            yield builder
        except BaseException:
            builder.status = "error"
            raise
        finally:
            builder.latency_ms = (self._clock() - start) * 1000.0
            stack.pop()
            span = builder.build(self._cost_model)
            if listener is not None:
                listener.on_close(span)
            if parent is not None:
                parent.children.append(span)
            else:
                self.last_trace = span
                self._emit(span)
            if token is not None:
                _stack.reset(token)

    def _emit(self, root: Span) -> None:
        """Log the finished trace, then fan it out to every exporter."""
        self._log(root)
        for sink in self._sinks:
            try:
                sink(root)
            except Exception:
                # A failing exporter must never surface to the request path.
                logger.exception("trace exporter %r failed", getattr(sink, "__name__", sink))

    def _log(self, root: Span) -> None:
        """One log line per finished trace."""
        total = root.total_usage
        logger.info(
            "trace name=%s status=%s latency_ms=%.1f tokens_in=%d tokens_out=%d usd=%.4f",
            root.name,
            root.status,
            root.latency_ms,
            total.tokens_in,
            total.tokens_out,
            root.total_usd,
        )


class _NullHandle:
    """A span handle that records nothing."""

    def record_usage(self, usage: Usage, model: str) -> None:
        """Discard the usage; this handle observes nothing."""

    def set(self, **attributes: str | int | float | bool) -> None:
        """Discard the attributes; this handle observes nothing."""


class NullTracer:
    """A :class:`~finrag.core.interfaces.Tracer` that does nothing.

    The default for services when no tracer is injected, so instrumented call
    sites stay uniform (``with tracer.span(...)``) without forcing every caller —
    tests, the eval runner — to wire up real tracing.
    """

    @contextmanager
    def span(self, name: str, **attributes: str | int | float | bool) -> Iterator[_NullHandle]:
        """Yield a no-op span handle."""
        yield _NullHandle()
