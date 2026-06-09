"""Generation-plane adapters.

LLM backends and answer-quality harness steps live here. Import each adapter
module here so its ``@register`` decorator runs.
"""

from finrag.generation import ollama  # imports register the adapter

__all__ = ["ollama"]
