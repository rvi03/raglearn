"""Tests for the Redis trace exporter (hermetic: fake publisher)."""

from __future__ import annotations

import json

from finrag.core.types import CostBreakdown, Span, Usage
from finrag.observability.redis_exporter import RedisSpanExporter


class _FakePublisher:
    """Captures the (channel, message) pairs published to it."""

    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []

    def publish(self, channel: str, message: str) -> int:
        self.published.append((channel, message))
        return 1


def _trace() -> Span:
    return Span(
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


def test_publishes_tree_and_totals_to_default_channel() -> None:
    pub = _FakePublisher()
    exporter = RedisSpanExporter(pub)

    exporter(_trace())

    assert len(pub.published) == 1
    channel, message = pub.published[0]
    assert channel == "finrag:trace"  # no trace_id → shared channel
    payload = json.loads(message)
    assert payload["name"] == "query"
    assert payload["attributes"] == {"path": "narrative"}
    assert payload["children"][0]["name"] == "generate"
    assert payload["children"][0]["model"] == "qwen2.5"
    # Convenience roll-ups for the root.
    assert payload["total_usd"] == 0.15
    assert payload["total_tokens_in"] == 12
    assert payload["total_tokens_out"] == 8


def test_per_trace_id_channel_when_present() -> None:
    pub = _FakePublisher()
    exporter = RedisSpanExporter(pub, channel_prefix="t")
    root = Span(name="query", attributes={"trace_id": "abc123"})

    exporter(root)

    channel, _message = pub.published[0]
    assert channel == "t:abc123"
