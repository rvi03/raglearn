"""Live token streaming seam.

A generation backend emits each token delta through this contextvar sink as it
arrives, so a consumer — the ``/chat`` SSE stream — can relay tokens live without
the backend knowing anything about HTTP. When no sink is installed (the plain
``/query`` path), emitting is a no-op and generation just returns its full text.

Contextvar-scoped like the span listener, so it isolates concurrent requests and
is carried into worker threads by ``asyncio.to_thread``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

_token_sink: ContextVar[Callable[[str], None] | None] = ContextVar(
    "finrag_token_sink", default=None
)


def set_token_sink(sink: Callable[[str], None]) -> Token[Callable[[str], None] | None]:
    """Install a per-request token sink; returns a token to reset it."""
    return _token_sink.set(sink)


def reset_token_sink(token: Token[Callable[[str], None] | None]) -> None:
    """Remove a previously installed token sink."""
    _token_sink.reset(token)


def emit_token(delta: str) -> None:
    """Hand one token delta to the installed sink, if any."""
    sink = _token_sink.get()
    if sink is not None and delta:
        sink(delta)


@contextmanager
def suppressed() -> Iterator[None]:
    """Mute token streaming for the duration of the block.

    Internal LLM calls — the groundedness judge, a query rewrite — must not leak
    their tokens into the user-facing answer stream, so they run inside this.
    """
    token = _token_sink.set(None)
    try:
        yield
    finally:
        _token_sink.reset(token)
