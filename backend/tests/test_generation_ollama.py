"""Tests for the Ollama LLM backend (hermetic via a mock transport)."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from finrag.core.errors import GenerationError
from finrag.core.registry import registry
from finrag.generation.ollama import OllamaBackend
from finrag.generation.token_stream import reset_token_sink, set_token_sink

Handler = Callable[[httpx.Request], httpx.Response]


def _backend(handler: Handler) -> OllamaBackend:
    backend = OllamaBackend(url="http://ollama:11434")
    backend._client = httpx.Client(transport=httpx.MockTransport(handler))
    return backend


def _ndjson(*chunks: dict[str, object]) -> httpx.Response:
    """A streaming NDJSON response, one JSON object per line (Ollama's format)."""
    body = "".join(json.dumps(chunk) + "\n" for chunk in chunks)
    return httpx.Response(200, content=body.encode())


def test_generate_assembles_text_usage_and_model() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _ndjson(
            {"response": "hel"},
            {"response": "lo"},
            {"done": True, "prompt_eval_count": 12, "eval_count": 5, "model": "qwen2.5"},
        )

    result = _backend(handler).generate("hi")

    assert result.text == "hello"  # deltas concatenated
    assert result.usage.tokens_in == 12
    assert result.usage.tokens_out == 5
    assert result.model == "qwen2.5"


def test_generate_emits_token_deltas_to_the_sink() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _ndjson({"response": "hel"}, {"response": "lo"}, {"done": True})

    deltas: list[str] = []
    token = set_token_sink(deltas.append)
    try:
        _backend(handler).generate("hi")
    finally:
        reset_token_sink(token)

    assert deltas == ["hel", "lo"]  # streamed live, in order


def test_generate_defaults_usage_when_absent() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _ndjson({"response": "x"}, {"done": True})

    result = _backend(handler).generate("hi")
    assert result.usage.tokens_in == 0
    assert result.usage.tokens_out == 0


def test_request_is_streaming_with_prompt_and_model() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _ndjson({"response": "ok"}, {"done": True})

    _backend(handler).generate("classify this")

    assert seen["stream"] is True
    assert seen["prompt"] == "classify this"
    assert "model" in seen


def test_http_error_raises_generation_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(GenerationError, match="ollama generate failed"):
        _backend(handler).generate("hi")


def test_registered() -> None:
    assert isinstance(registry.create("llm_backend", "ollama", url="http://x"), OllamaBackend)
