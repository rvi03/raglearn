"""Ingestion-plane adapters.

Connectors, metadata extractors, intake/dedup, parsers, chunkers, and embedders
live here. Import each adapter module here so its ``@register`` decorator runs.
"""

from raglearn.ingestion import detect  # re-exported; the import registers the adapter

__all__ = ["detect"]
