"""Generation-plane interfaces.

The LLM serving backend used to produce answers. The verification steps that wrap
generation live in the harness package.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from finrag.core.types import LLMResponse


@runtime_checkable
class LLMBackend(Protocol):
    """Serves the text LLM (Ollama by default; vLLM/TGI when GPU)."""

    def generate(self, prompt: str) -> LLMResponse:
        """Return the model's completion and token usage for a prompt."""
        ...
