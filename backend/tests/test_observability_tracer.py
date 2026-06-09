"""Tests for the in-process tracer: nesting, timing, and cost roll-up."""

from __future__ import annotations

import pytest

from finrag.core.types import CostBreakdown, Span, Usage
from finrag.observability.tracer import InProcessTracer


class _FakeClock:
    """A monotonic clock the test advances by hand, in seconds."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _PennyPerToken:
    """A CostModel that prices every token at $0.01, for assertable totals."""

    def price(self, usage: Usage, model: str) -> CostBreakdown:
        return CostBreakdown(
            model=model,
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
            usd=0.01 * (usage.tokens_in + usage.tokens_out),
        )


def _tracer(clock: _FakeClock) -> InProcessTracer:
    return InProcessTracer(cost_model=_PennyPerToken(), clock=clock)


def test_sinks_receive_the_finished_root_and_failures_are_swallowed() -> None:
    clock = _FakeClock()
    received: list[Span] = []
    bad_called: list[bool] = []

    def good(root: Span) -> None:
        received.append(root)

    def bad(root: Span) -> None:
        bad_called.append(True)
        raise RuntimeError("exporter down")

    # The failing sink is registered first; the good one must still run, and the
    # traced block itself must complete normally.
    tracer = InProcessTracer(cost_model=_PennyPerToken(), clock=clock, sinks=[bad, good])

    with tracer.span("query"):
        pass

    assert bad_called == [True]  # the failing sink ran
    assert len(received) == 1 and received[0].name == "query"  # and did not stop the next


def test_nesting_builds_a_tree() -> None:
    clock = _FakeClock()
    tracer = _tracer(clock)

    with tracer.span("query"):
        with tracer.span("retrieve"):
            pass
        with tracer.span("generate"):
            pass

    root = tracer.last_trace
    assert root is not None
    assert root.name == "query"
    assert [c.name for c in root.children] == ["retrieve", "generate"]


def test_latency_is_measured_per_span() -> None:
    clock = _FakeClock()
    tracer = _tracer(clock)

    with tracer.span("query"):
        with tracer.span("retrieve"):
            clock.advance(0.2)  # 200 ms inside retrieve
        clock.advance(0.05)  # 50 ms more inside query

    root = tracer.last_trace
    assert root is not None
    assert root.children[0].latency_ms == pytest.approx(200.0)
    assert root.latency_ms == pytest.approx(250.0)


def test_usage_is_priced_and_rolled_up() -> None:
    clock = _FakeClock()
    tracer = _tracer(clock)

    with tracer.span("query"):
        with tracer.span("generate") as gen:
            gen.record_usage(Usage(tokens_in=10, tokens_out=5), model="qwen2.5")

    root = tracer.last_trace
    assert root is not None
    gen_span = root.children[0]
    # The leaf span carries the priced call.
    assert gen_span.cost is not None
    assert gen_span.cost.usd == pytest.approx(0.15)
    assert gen_span.model == "qwen2.5"
    # The root has no LLM call of its own, but rolls up the subtree.
    assert root.cost is None
    assert root.total_usd == pytest.approx(0.15)
    assert root.total_usage == Usage(tokens_in=10, tokens_out=5)


def test_usage_sums_across_calls_on_one_span() -> None:
    clock = _FakeClock()
    tracer = _tracer(clock)

    with tracer.span("generate") as gen:
        gen.record_usage(Usage(tokens_in=10, tokens_out=5), model="qwen2.5")
        gen.record_usage(Usage(tokens_in=3, tokens_out=2), model="qwen2.5")

    root = tracer.last_trace
    assert root is not None
    assert root.usage == Usage(tokens_in=13, tokens_out=7)
    assert root.total_usd == pytest.approx(0.20)


def test_error_in_block_marks_span_and_propagates() -> None:
    clock = _FakeClock()
    tracer = _tracer(clock)

    with pytest.raises(ValueError):
        with tracer.span("query"):
            with tracer.span("retrieve"):
                raise ValueError("boom")

    root = tracer.last_trace
    assert root is not None
    # The child that raised is marked; because the error propagates through the
    # enclosing block, the parent span is marked failed too.
    assert root.children[0].status == "error"
    assert root.status == "error"


def test_attributes_are_attached() -> None:
    clock = _FakeClock()
    tracer = _tracer(clock)

    with tracer.span("retrieve", top_k=8) as ret:
        ret.set(reranked=True)

    root = tracer.last_trace
    assert root is not None
    assert root.attributes == {"top_k": 8, "reranked": True}


def test_roll_up_properties_on_a_hand_built_tree() -> None:
    # The roll-up is a plain property on Span, independent of the tracer.
    tree = Span(
        name="query",
        children=[
            Span(
                name="a",
                usage=Usage(tokens_in=1, tokens_out=1),
                cost=CostBreakdown(model="m", tokens_in=1, tokens_out=1, usd=0.5),
            ),
            Span(
                name="b",
                usage=Usage(tokens_in=2, tokens_out=0),
                cost=CostBreakdown(model="m", tokens_in=2, tokens_out=0, usd=0.25),
            ),
        ],
    )
    assert tree.total_usd == pytest.approx(0.75)
    assert tree.total_usage == Usage(tokens_in=3, tokens_out=1)
