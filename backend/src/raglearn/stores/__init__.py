"""Storage-plane adapters and clients.

Vector store (Qdrant), structured store (DuckDB), and the graph index live here.
Import each adapter module here so its ``@register`` decorator runs.
"""

from raglearn.stores import duckdb_structured  # re-exported; the import registers the adapter

__all__ = ["duckdb_structured"]
