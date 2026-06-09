"""Entity resolution: pin a query to the company it is about.

Resolves the company named in the query to its collection and records it as a
``collection_id`` filter. That filter does double duty: it scopes the narrative
vector search to that company's chunks (instead of the whole corpus) and feeds
the exact path the collection it should read facts from.

Deterministic and cheap — a name/ticker match over the known collections, no LLM.
An explicit ``collection_id`` already on the query is left untouched, and an
unresolvable company simply leaves the query unscoped (graceful, not an error).
"""

from __future__ import annotations

from typing import Protocol

from finrag.core.types import Query


class _CollectionResolver(Protocol):
    """The slice of the structured store used to resolve a company."""

    def find_collection(self, text: str) -> tuple[str, str] | None: ...


def resolve_entity(query: Query, store: _CollectionResolver) -> Query:
    """Return the query with a ``collection_id`` filter for its company, if found.

    Args:
      query: The user question and its filters.
      store: Resolves a company name/ticker to ``(collection_id, company)``.

    Returns:
      The query with ``filters["collection_id"]`` set when a company resolves;
      otherwise the query unchanged (explicit filter or no match).
    """
    if "collection_id" in query.filters:
        return query
    found = store.find_collection(query.text)
    if found is None:
        return query
    return query.model_copy(update={"filters": {**query.filters, "collection_id": found[0]}})
