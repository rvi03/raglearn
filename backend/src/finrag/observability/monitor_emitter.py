"""Monitor emitters: publish ingestion progress for the live DAG.

The ingestion counterpart of the Redis trace exporter. As the consumer drives a
document through the pipeline, it calls a :class:`~finrag.core.interfaces.MonitorEmitter`
at each stage boundary; the Redis implementation publishes those events to a
single channel that a server-sent-events endpoint relays to the monitor view.

Thin by design: each emitter owns serialization + the channel, not the pipeline.
The default is a no-op, so ingestion runs unobserved unless an emitter is wired in
(and tests stay hermetic without one).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from finrag.core.interfaces.crosscutting import MonitorEmitter

logger = logging.getLogger(__name__)

# One channel for all uploads — the monitor view is a global feed, not per-job.
INGESTION_CHANNEL = "finrag:ingestion"


class CompositeMonitorEmitter:
    """Fans an event out to several emitters (e.g. durable Postgres + live Redis).

    A failure in one sink (a Redis blip, a transient DB error) must not drop the
    event for the others or break ingestion, so each delegate call is isolated and
    logged rather than raised.
    """

    def __init__(self, emitters: Sequence[MonitorEmitter]) -> None:
        """Bind the fan-out to its delegate emitters, in order."""
        self._emitters = list(emitters)

    def upload(
        self, *, upload_id: str, country: str, created: str, docs: Sequence[tuple[str, str]]
    ) -> None:
        """Fan out an ``upload`` event."""
        for e in self._emitters:
            try:
                e.upload(upload_id=upload_id, country=country, created=created, docs=docs)
            except Exception:
                logger.exception("monitor emitter %s failed on upload", type(e).__name__)

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
        """Fan out a ``node`` event."""
        for e in self._emitters:
            try:
                e.node(
                    upload_id=upload_id,
                    doc_id=doc_id,
                    stage=stage,
                    label=label,
                    status=status,
                    detail=detail,
                )
            except Exception:
                logger.exception("monitor emitter %s failed on node", type(e).__name__)

    def doc_done(self, *, upload_id: str, doc_id: str, outcome: str) -> None:
        """Fan out a ``doc_done`` event."""
        for e in self._emitters:
            try:
                e.doc_done(upload_id=upload_id, doc_id=doc_id, outcome=outcome)
            except Exception:
                logger.exception("monitor emitter %s failed on doc_done", type(e).__name__)


@runtime_checkable
class _Publisher(Protocol):
    """The slice of a Redis client this emitter needs."""

    def publish(self, channel: str, message: str) -> Any:
        """Publish ``message`` to ``channel``."""
        ...


class NullMonitorEmitter:
    """A :class:`~finrag.core.interfaces.MonitorEmitter` that does nothing."""

    def upload(
        self, *, upload_id: str, country: str, created: str, docs: Sequence[tuple[str, str]]
    ) -> None:
        """No-op."""

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
        """No-op."""

    def doc_done(self, *, upload_id: str, doc_id: str, outcome: str) -> None:
        """No-op."""


class RedisMonitorEmitter:
    """A monitor emitter that publishes events to Redis pub/sub.

    Each method serializes the event in the wire shape the monitor frontend reads
    (a ``type``-tagged JSON object) and publishes it to the shared channel.
    """

    def __init__(self, client: _Publisher, *, channel: str = INGESTION_CHANNEL) -> None:
        """Bind the emitter to a Redis publisher.

        Args:
          client: A Redis client (anything with ``publish(channel, message)``).
          channel: The pub/sub channel to publish ingestion events to.
        """
        self._client = client
        self._channel = channel

    def _publish(self, event: dict[str, Any]) -> None:
        self._client.publish(self._channel, json.dumps(event))

    def upload(
        self, *, upload_id: str, country: str, created: str, docs: Sequence[tuple[str, str]]
    ) -> None:
        """Publish the ``upload`` event announcing a batch and its documents."""
        self._publish(
            {
                "type": "upload",
                "upload_id": upload_id,
                "country": country,
                "created": created,
                "docs": [{"doc_id": doc_id, "filename": filename} for doc_id, filename in docs],
            }
        )

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
        """Publish a ``node`` event for one document's stage transition."""
        event: dict[str, Any] = {
            "type": "node",
            "upload_id": upload_id,
            "doc_id": doc_id,
            "id": stage,
            "label": label,
            "status": status,
        }
        if detail is not None:
            event["detail"] = detail
        self._publish(event)

    def doc_done(self, *, upload_id: str, doc_id: str, outcome: str) -> None:
        """Publish the ``doc_done`` event for one document's terminal outcome."""
        self._publish(
            {"type": "doc_done", "upload_id": upload_id, "doc_id": doc_id, "outcome": outcome}
        )
