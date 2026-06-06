"""Event-driven ingestion consumer.

Subscribes to the ingestion topic, and for every object-created notification:
decode the event, fetch the object's bytes, and hand the result to a processor.
The consumer is pure transport — it knows nothing about parsing or chunking. The
``Processor`` seam is where the rest of the ingestion pipeline attaches; the
default processor routes each document to its parser via :class:`ParserRouter`.

Run it directly for local development::

    uv run python -m raglearn.ingestion.consumer          # listen forever
    uv run python -m raglearn.ingestion.consumer --once   # handle one message, exit
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable

from aiokafka import AIOKafkaConsumer

from raglearn.core.config import Settings, load_settings
from raglearn.core.logging import configure_logging
from raglearn.core.types import RawDocument
from raglearn.ingestion.events import parse_object_events
from raglearn.ingestion.object_store import ObjectStore
from raglearn.ingestion.router import ParserRouter, build_router

logger = logging.getLogger(__name__)

# Receives one fetched document. The parse/chunk/embed pipeline plugs in here.
Processor = Callable[[RawDocument], Awaitable[None]]

_GROUP_ID = "raglearn-ingest-consumer"


def routed_processor(router: ParserRouter) -> Processor:
    """Adapt a synchronous router into an async processor for the consumer.

    The router's detect-and-parse work is blocking (HTTP detect, CPU parsing),
    so it runs in a worker thread to keep the consumer's event loop free.

    Args:
      router: The parse router to drive each document through.

    Returns:
      An async processor that routes one document per call.
    """

    async def process(document: RawDocument) -> None:
        await asyncio.to_thread(router.process, document)

    return process


class IngestionConsumer:
    """Drains object-created events from the ingestion topic."""

    def __init__(self, settings: Settings, store: ObjectStore, process: Processor) -> None:
        """Wire the consumer to its broker, object store, and downstream processor.

        Args:
          settings: Application settings (broker address and topic name).
          store: Source of object bytes.
          process: Downstream handler for each fetched document.
        """
        self._topic = settings.services.ingest_topic
        self._broker = settings.services.redpanda_broker
        self._store = store
        self._process = process

    async def run(self, *, once: bool = False) -> None:
        """Consume events until cancelled.

        Args:
          once: If set, handle a single message and return — a development aid.
        """
        consumer = AIOKafkaConsumer(
            self._topic,
            bootstrap_servers=self._broker,
            group_id=_GROUP_ID,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        )
        await consumer.start()
        logger.info("listening on %s via %s", self._topic, self._broker)
        try:
            async for message in consumer:
                await self._handle(message.value)
                await consumer.commit()
                if once:
                    return
        finally:
            await consumer.stop()

    async def _handle(self, value: bytes | None) -> None:
        """Decode one notification and process each object it references."""
        if value is None:
            return
        for event in parse_object_events(value):
            data = await self._store.fetch(event.bucket, event.key)
            document = RawDocument(
                doc_id=event.key,
                filename=event.key.rsplit("/", 1)[-1],
                content_type=event.content_type or "application/octet-stream",
                data=data,
                source_bucket=event.bucket,
            )
            await self._process(document)


async def _main(once: bool) -> None:
    """Build a consumer from settings and run it with the default processor."""
    settings = load_settings()
    store = ObjectStore(
        settings.services.minio_endpoint,
        settings.services.minio_access_key,
        settings.services.minio_secret_key,
    )
    process = routed_processor(build_router(settings, store))
    await IngestionConsumer(settings, store, process).run(once=once)


if __name__ == "__main__":
    configure_logging()
    asyncio.run(_main(once="--once" in sys.argv[1:]))
