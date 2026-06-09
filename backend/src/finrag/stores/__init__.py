"""Storage-plane adapters and clients.

Vector store (Qdrant), structured store (DuckDB or Postgres), and the graph index
live here. Import each adapter module here so its ``@register`` decorator runs.
"""

from finrag.stores import (  # imports register the adapters
    duckdb_structured,
    postgres_structured,
    qdrant_vector,
)

__all__ = ["duckdb_structured", "postgres_structured", "qdrant_vector"]
