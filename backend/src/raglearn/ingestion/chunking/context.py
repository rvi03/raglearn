"""Deterministic context prefixing for chunk embedding.

A bare chunk often loses what document and section it came from, which hurts
retrieval. Prepending a short, deterministic context line -- the filing's
identity and the chunk's section -- recovers most of that signal at no cost,
unlike an LLM-generated per-chunk summary. The prefix is built from metadata at
embed time and is not stored on the chunk, so the chunk's own text stays clean
for display and citation.
"""

from __future__ import annotations

from raglearn.core.types import Chunk

_SEP = " · "


def context_prefix(chunk: Chunk) -> str:
    """Return the deterministic context line for a chunk, or ``""`` if empty.

    Built from the chunk's binding metadata: company, filing type, fiscal
    period (or year), and section heading -- whichever are present.
    """
    meta = chunk.metadata
    period = meta.fiscal_period or (str(meta.fiscal_year) if meta.fiscal_year is not None else None)
    parts = [meta.company_name, meta.filing_type, period, meta.section]
    return _SEP.join(part for part in parts if part)


def contextualize(chunk: Chunk) -> str:
    """Return the chunk's text with its context line prepended for embedding."""
    prefix = context_prefix(chunk)
    return f"{prefix}\n\n{chunk.text}" if prefix else chunk.text
