"""Ingestion-plane adapters.

Connectors, metadata extractors, intake/dedup, parsers, chunkers, and embedders
live here. Import each adapter module here so its ``@register`` decorator runs.
"""

from finrag.ingestion import detect, embedding, identity  # imports register the adapters

__all__ = ["detect", "embedding", "identity"]
