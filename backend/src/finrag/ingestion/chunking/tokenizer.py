"""Token counting aligned to the embedding model.

Chunk sizes are measured in the same tokens the embedder will see, so a chunk
that fits the budget here fits the model at embed time. bge-m3's own tokenizer
is the reference; it is loaded lazily and cached, since the heavy import and the
one-time tokenizer download should not happen at module import or in tests that
inject their own counter.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

TokenCounter = Callable[[str], int]

_BGE_M3_MODEL = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def _bge_m3_tokenizer() -> object:
    """Load and cache bge-m3's tokenizer (lazy: imports transformers on first use)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(_BGE_M3_MODEL)  # type: ignore[no-untyped-call]


def bge_m3_token_counter() -> TokenCounter:
    """Return a counter giving a string's length in bge-m3 tokens.

    Special tokens are excluded so the count reflects content length, which is
    what the chunk-size budget is about.
    """
    tokenizer = _bge_m3_tokenizer()

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))  # type: ignore[attr-defined]

    return count
