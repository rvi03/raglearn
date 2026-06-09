"""Monitor endpoint: stream the live ingestion DAG (SSE).

The ingestion counterpart of ``/chat``'s live trace. The consumer publishes
``upload``/``node``/``doc_done`` events to a Redis channel as it drives documents
through the pipeline; this endpoint subscribes to that channel and relays each
event to the monitor view as it happens.

Wire format matches the rest of the surface: ``data: {json}`` frames whose
``type`` is inside the payload (the events are already in that shape on the
channel, so they are relayed verbatim). It is a single global feed of all uploads,
not a per-job stream.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from finrag.api.deps import get_monitor_store
from finrag.core.logging import get_logger
from finrag.observability import INGESTION_CHANNEL
from finrag.stores.postgres_monitor import PostgresMonitorStore

logger = get_logger(__name__)
router = APIRouter(tags=["monitor"])

# Idle poll interval. We poll the subscription rather than block on ``listen()``:
# a blocking read times out on an idle channel (the connection's socket timeout),
# which would crash the stream. On each idle tick we emit an SSE comment frame so
# the connection stays open through proxies and we notice client disconnects.
_KEEPALIVE_S = 15.0


async def _subscribe(redis_url: str) -> AsyncIterator[str]:
    """Yield ingestion events from the Redis channel as SSE ``data:`` frames."""
    import redis.asyncio as aioredis

    client = aioredis.from_url(redis_url, decode_responses=True)
    pubsub = client.pubsub()
    await pubsub.subscribe(INGESTION_CHANNEL)
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=_KEEPALIVE_S)
            if message is None:
                yield ": keepalive\n\n"  # comment frame — holds the connection open while idle
                continue
            # The payload is already a type-tagged JSON event; relay verbatim.
            yield f"data: {message['data']}\n\n"
    finally:
        await pubsub.unsubscribe(INGESTION_CHANNEL)
        await pubsub.aclose()  # type: ignore[no-untyped-call]
        await client.aclose()


@router.get("/ingestion/events")
async def ingestion_events(request: Request) -> StreamingResponse:
    """Stream live ingestion-pipeline events for the monitor DAG."""
    redis_url = request.app.state.settings.services.redis_url
    return StreamingResponse(_subscribe(redis_url), media_type="text/event-stream")


@router.get("/ingestion/uploads")
async def ingestion_uploads(
    store: Annotated[PostgresMonitorStore, Depends(get_monitor_store)],
) -> dict[str, list[dict[str, Any]]]:
    """List recent uploads with per-document ingestion status (durable).

    The persistent counterpart of ``/ingestion/events`` — the corpus and monitor
    views read this so they show uploads and their status at any time, not only
    while a run is streaming live.
    """
    return {"uploads": store.list_uploads()}
