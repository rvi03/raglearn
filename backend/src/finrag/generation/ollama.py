"""Ollama text-LLM backend.

The default local serving path: a self-hosted Ollama exposing ``/api/generate``.
Non-streaming (one request, one completion) since callers here want the whole
answer plus token counts for cost accounting. vLLM/TGI are alternative backends
behind the same :class:`~finrag.core.interfaces.LLMBackend` interface.
"""

from __future__ import annotations

import json

import httpx

from finrag.core.errors import GenerationError
from finrag.core.registry import registry
from finrag.core.types import LLMResponse, Usage
from finrag.generation.token_stream import emit_token

# Qwen2.5-7B-Instruct is the default text model; overridable per deployment.
_DEFAULT_MODEL = "qwen2.5:7b-instruct"


@registry.register("llm_backend", "ollama")
class OllamaBackend:
    """An :class:`~finrag.core.interfaces.LLMBackend` served by a local Ollama."""

    def __init__(self, url: str, *, model: str = _DEFAULT_MODEL, timeout_s: float = 120.0) -> None:
        """Bind the backend to an Ollama server.

        Args:
          url: Base URL of the Ollama server, e.g. ``http://localhost:11434``.
          model: The model tag to generate with.
          timeout_s: Per-request timeout in seconds (local CPU generation is slow).
        """
        self._url = url.rstrip("/") + "/api/generate"
        self._model = model
        self._client = httpx.Client(timeout=timeout_s)

    def generate(self, prompt: str) -> LLMResponse:
        """Return the model's completion and token usage for a prompt.

        Streams the completion from Ollama: each token delta is forwarded to the
        live token sink (so ``/chat`` can relay it as it arrives), while the full
        text is accumulated and returned. Callers that do not install a sink (the
        plain ``/query`` path) just receive the assembled response.

        Raises:
          GenerationError: The Ollama server could not be reached or errored.
        """
        text_parts: list[str] = []
        tokens_in = 0
        tokens_out = 0
        model = self._model
        try:
            with self._client.stream(
                "POST",
                self._url,
                json={"model": self._model, "prompt": prompt, "stream": True},
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    delta = chunk.get("response", "")
                    if delta:
                        text_parts.append(delta)
                        emit_token(delta)
                    # The final chunk (``done``) carries the token counts and model.
                    if chunk.get("done"):
                        tokens_in = int(chunk.get("prompt_eval_count", 0))
                        tokens_out = int(chunk.get("eval_count", 0))
                        model = chunk.get("model", self._model)
        except httpx.HTTPError as exc:
            raise GenerationError(f"ollama generate failed: {exc}") from exc
        return LLMResponse(
            text="".join(text_parts),
            usage=Usage(tokens_in=tokens_in, tokens_out=tokens_out),
            model=model,
        )
