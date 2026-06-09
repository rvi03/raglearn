"""Chat endpoint: stream a grounded answer and its live agent trace (SSE).

The streaming counterpart of ``/query``. It emits the event contract the
frontend reads: ``agent_step`` events stream **live** as each pipeline stage runs
(``running`` on open, ``done``/``failed`` on close), then the evidence
(``source``/``citation``), the answer (``token``), and a final ``done`` carrying
the query's token usage and cost.

Mechanics: the query path (``QueryService.answer``) is synchronous and blocking
(embedding, reranking, LLM generation), so it runs in a worker thread. A
per-request :class:`SpanListener` — propagated into that thread because
``asyncio.to_thread`` copies the context — pushes step events onto an async
queue as spans open and close, which the SSE generator relays as they happen.

Wire format: frames are ``data: {json}`` with the event ``type`` inside the
payload (``finrag.api.sse.format_data_event``), which is what the frontend's
SSE reader discriminates on.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from finrag.api.deps import (
    get_chat_store,
    get_output_guard,
    get_pii_redactor,
    get_query_service,
)
from finrag.api.sse import format_data_event
from finrag.core.interfaces.crosscutting import OutputGuard, PiiRedactor
from finrag.core.logging import get_logger
from finrag.core.types import Citation, Query, Span, Usage
from finrag.generation.token_stream import reset_token_sink, set_token_sink
from finrag.observability.tracer import (
    reset_span_listener,
    reset_trace_id,
    set_span_listener,
    set_trace_id,
)
from finrag.retrieval.answer import GroundedAnswer, SourceCard
from finrag.retrieval.query_service import QueryService
from finrag.security.output_stream import SegmentScreener
from finrag.stores.postgres_chat import PostgresChatStore

logger = get_logger(__name__)
router = APIRouter(tags=["chat"])

# Shown as the answer when the pipeline errors mid-stream.
_STREAM_ERROR = "Something went wrong answering this question."
# Appended live when the output guard blocks a segment mid-stream.
_STREAM_BLOCKED = " [answer withheld — failed a safety check]"
# Longest auto-title taken from a conversation's first message.
_TITLE_MAX = 60


class ChatRequest(BaseModel):
    """A question to answer, streamed."""

    text: str
    # When set, the turn is persisted to this conversation and its recent turns
    # are fed back as short-term memory (history-aware rewrite). Omit for a
    # one-shot, stateless answer.
    session_id: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    # PLACEHOLDER for auth-derived tags; never trust access scope from the client.
    access_tags: list[str] = Field(default_factory=list)


def _auto_title(text: str) -> str:
    """Derive a conversation title from its first message (truncated)."""
    title = " ".join(text.split())
    return title[: _TITLE_MAX - 1] + "…" if len(title) > _TITLE_MAX else title


def _source_dicts(cards: list[SourceCard]) -> list[dict[str, object]]:
    """Source cards in the SSE wire shape (also what a reopened turn replays)."""
    return [{"id": c.id, "title": c.title, "url": c.url, "snippet": c.snippet} for c in cards]


def _citation_dicts(citations: list[Citation]) -> list[dict[str, object]]:
    """Citations in the SSE wire shape (also what a reopened turn replays)."""
    return [
        {"id": c.id, "source_doc_id": c.source_doc_id, "page": c.page or 0, "span": c.section or ""}
        for c in citations
    ]


def _step_detail(span: Span) -> str:
    """Build a detail string from a span's decision attributes.

    The attributes carry the actual decision context (``hits``, ``arms``,
    ``fused``, ``top_k``, ``kept``, ``candidate_k``, ``allowed``, ``entities``…);
    ``path`` and ``question`` are surfaced elsewhere, so they are excluded.
    """
    return ", ".join(
        f"{k}={v}" for k, v in span.attributes.items() if k not in ("path", "question")
    )


def _agent_step(span: Span) -> str:
    """Render a finished span as a ``done``/``failed`` agent_step frame."""
    return format_data_event(
        "agent_step",
        {
            "name": span.name,
            "path": str(span.attributes.get("path", "")),
            "detail": _step_detail(span),
            "status": "done" if span.status == "ok" else "failed",
            "latency_ms": span.latency_ms,
            "tokens_in": span.usage.tokens_in,
            "tokens_out": span.usage.tokens_out,
            "model": span.model or "",
            "usd": span.cost.usd if span.cost is not None else 0.0,
        },
    )


def _persist_user_turn(store: PostgresChatStore, session_id: str, text: str) -> None:
    """Create the conversation (first turn names it) and append the user message."""
    store.create_session(session_id=session_id, title=_auto_title(text))
    store.append_message(message_id=uuid4().hex, session_id=session_id, role="user", text=text)


class _QueueListener:
    """A :class:`SpanListener` that pushes step frames onto an async queue.

    ``on_open``/``on_close`` run in the worker thread, so they hand frames back to
    the event loop with ``call_soon_threadsafe``. The last span to close is the
    root, kept for the query's total usage and cost.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[str | None]) -> None:
        self._loop = loop
        self._queue = queue
        self.root: Span | None = None

    def _put(self, frame: str) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, frame)

    def on_open(self, name: str, attributes: dict[str, str | int | float | bool]) -> None:
        self._put(
            format_data_event(
                "agent_step",
                {
                    "name": name,
                    "path": str(attributes.get("path", "")),
                    "status": "running",
                    "latency_ms": 0.0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "model": "",
                    "usd": 0.0,
                },
            )
        )

    def on_close(self, span: Span) -> None:
        self.root = span  # spans close inner-first, so the last close is the root
        self._put(_agent_step(span))


async def _event_stream(
    service: QueryService,
    query: Query,
    output_guard: OutputGuard,
    pii_redactor: PiiRedactor,
    store: PostgresChatStore | None = None,
    session_id: str | None = None,
) -> AsyncIterator[str]:
    """Yield the SSE event frames for one streamed answer.

    When ``store`` and ``session_id`` are set, the conversation's recent turns are
    read back as short-term memory (fed to the history-aware rewrite) and this
    exchange is persisted: the user turn before generation, the assistant turn
    after. DB calls run off the event loop (the pool is sync).
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    listener = _QueueListener(loop, queue)
    # A 32-hex trace id: the client gets it on `done`, and the Langfuse exporter
    # anchors its OTel trace to it, so the two line up for a deep link.
    trace_id = uuid4().hex
    streamed = {"any": False}

    # Short-term memory: fold the conversation's recent turns into the query so the
    # rewrite can resolve coreference ("its margins?"). Read *before* persisting the
    # current turn so this question is not in its own history. Then record the user
    # turn (which also names the conversation on its first message).
    if store is not None and session_id is not None:
        history = await asyncio.to_thread(store.recent_turns, session_id=session_id)
        query = query.model_copy(update={"history": history})
        await asyncio.to_thread(_persist_user_turn, store, session_id, query.text)

    def on_token(delta: str) -> None:
        streamed["any"] = True
        loop.call_soon_threadsafe(queue.put_nowait, format_data_event("token", {"delta": delta}))

    def on_block() -> None:
        # A segment failed the output guard mid-stream: note it live. The
        # whole-answer screen returns the refusal, so don't re-emit it here.
        streamed["any"] = True
        loop.call_soon_threadsafe(
            queue.put_nowait, format_data_event("token", {"delta": _STREAM_BLOCKED})
        )

    # Screen and redact each segment live with the same guard and redactor the
    # whole-answer checks use, so neither an unsafe answer nor PII can reach the
    # user before the final screen runs.
    screener = SegmentScreener(
        guard=output_guard, downstream=on_token, on_block=on_block, redactor=pii_redactor
    )

    def finalize() -> None:
        # On the loop thread after the worker finishes: release any trailing
        # buffered segment, then close the stream. The sentinel is scheduled after
        # the flushed token frames (FIFO), so it always arrives last.
        screener.flush()
        loop.call_soon_threadsafe(queue.put_nowait, None)

    span_token = set_span_listener(listener)
    sink_token = set_token_sink(screener)
    tid_token = set_trace_id(trace_id)
    try:
        # Run the blocking pipeline off the event loop; the copied context carries
        # the listener, token sink, and trace id so they apply inside the thread.
        future = asyncio.ensure_future(asyncio.to_thread(service.answer, query))
        future.add_done_callback(lambda _f: finalize())

        while True:
            frame = await queue.get()
            if frame is None:  # sentinel: the pipeline has finished
                break
            yield frame  # live agent_step and token frames, in order

        answer: GroundedAnswer = future.result()  # re-raises a pipeline failure
        sources = _source_dicts(answer.sources)
        citations = _citation_dicts(answer.citations)
        for source in sources:
            yield format_data_event("source", source)
        for citation in citations:
            yield format_data_event("citation", citation)
        # The answer streamed token-by-token during generation; emit it whole only
        # when nothing streamed (e.g. an abstention that never called the LLM).
        if not streamed["any"]:
            yield format_data_event("token", {"delta": answer.answer})

        usage = listener.root.total_usage if listener.root is not None else Usage()
        query_usd = listener.root.total_usd if listener.root is not None else 0.0
        yield format_data_event(
            "done",
            {
                "usage": {"tokens_in": usage.tokens_in, "tokens_out": usage.tokens_out},
                "trace_id": trace_id,
                "query_usd": query_usd,
                "grounding_confidence": answer.grounding_confidence,
                # Safety/privacy signals for the UI: whether the answer was refused
                # by a guard, and which PII types were masked.
                "answered": answer.answered,
                "redacted": answer.redacted,
            },
        )

        # Persist the assistant turn (same wire shapes the UI replays on reopen).
        # Guarded on its own so a storage hiccup never re-emits a stream error.
        if store is not None and session_id is not None:
            meta: dict[str, object] = {
                "sources": sources,
                "citations": citations,
                "costUsd": query_usd,
                "traceId": trace_id,
                "groundingConfidence": answer.grounding_confidence,
                "answered": answer.answered,
                "redacted": answer.redacted,
            }
            try:
                await asyncio.to_thread(
                    store.append_message,
                    message_id=uuid4().hex,
                    session_id=session_id,
                    role="assistant",
                    text=answer.answer,
                    meta=meta,
                )
            except Exception:
                logger.exception("failed to persist assistant turn")
    except Exception:
        logger.exception("chat stream failed")
        yield format_data_event("token", {"delta": _STREAM_ERROR})
        yield format_data_event(
            "done",
            {"usage": {"tokens_in": 0, "tokens_out": 0}, "trace_id": trace_id, "query_usd": 0.0},
        )
    finally:
        reset_span_listener(span_token)
        reset_token_sink(sink_token)
        reset_trace_id(tid_token)


@router.post("/chat")
async def chat(
    request: ChatRequest,
    http_request: Request,
    service: Annotated[QueryService, Depends(get_query_service)],
    output_guard: Annotated[OutputGuard, Depends(get_output_guard)],
    pii_redactor: Annotated[PiiRedactor, Depends(get_pii_redactor)],
) -> StreamingResponse:
    """Answer a question, streaming the live agent trace and the cited answer.

    With a ``session_id`` the turn joins a persistent conversation (short-term
    memory + history); without one it is answered statelessly. The chat store is
    opened only when a session is in play, so a one-shot answer needs no database.
    """
    store = get_chat_store(http_request) if request.session_id else None
    query = Query(text=request.text, filters=request.filters, access_tags=request.access_tags)
    return StreamingResponse(
        _event_stream(
            service,
            query,
            output_guard,
            pii_redactor,
            store=store,
            session_id=request.session_id,
        ),
        media_type="text/event-stream",
    )
