"""Redis pub/sub trace exporter.

Publishes a finished trace (the root :class:`~finrag.core.types.Span` and its
subtree) as JSON to a Redis channel. A server-sent-events endpoint subscribes to
that channel and relays the agent-trace/DAG to the UI live — the same
``{stage, status, latency, tokens, usd}`` shape the chat surface renders.

The exporter is deliberately thin: it owns serialization and the channel name,
not the transport. It is handed a publisher (a Redis client) so it stays
hermetically testable with a fake.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from finrag.core.types import Span

# A trace whose root span carries this attribute is published to a per-trace
# channel (``<prefix>:<id>``) so a subscriber can scope to one query; without it
# the trace goes to the shared prefix channel.
_TRACE_ID_ATTR = "trace_id"
_DEFAULT_CHANNEL_PREFIX = "finrag:trace"


@runtime_checkable
class _Publisher(Protocol):
    """The slice of a Redis client this exporter needs."""

    def publish(self, channel: str, message: str) -> Any:
        """Publish ``message`` to ``channel``."""
        ...


def _payload(root: Span) -> dict[str, Any]:
    """Serialize the trace tree plus convenience roll-ups for the root."""
    total = root.total_usage
    return {
        **root.model_dump(),
        "total_usd": root.total_usd,
        "total_tokens_in": total.tokens_in,
        "total_tokens_out": total.tokens_out,
    }


class RedisSpanExporter:
    """A trace sink that publishes finished traces to Redis pub/sub."""

    def __init__(
        self, client: _Publisher, *, channel_prefix: str = _DEFAULT_CHANNEL_PREFIX
    ) -> None:
        """Bind the exporter to a Redis publisher.

        Args:
          client: A Redis client (anything with ``publish(channel, message)``).
          channel_prefix: Base channel; a per-trace id (if present on the root
            span) is appended as ``<prefix>:<id>``.
        """
        self._client = client
        self._channel_prefix = channel_prefix

    def __call__(self, root: Span) -> None:
        """Publish the finished trace as JSON."""
        trace_id = root.attributes.get(_TRACE_ID_ATTR)
        channel = (
            f"{self._channel_prefix}:{trace_id}" if trace_id is not None else self._channel_prefix
        )
        self._client.publish(channel, json.dumps(_payload(root)))
