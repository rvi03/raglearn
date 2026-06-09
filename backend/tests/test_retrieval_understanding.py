"""Tests for entity resolution (query → company collection filter)."""

from __future__ import annotations

from finrag.core.types import Query
from finrag.retrieval.understanding import resolve_entity


class _Store:
    def __init__(self, found: tuple[str, str] | None) -> None:
        self._found = found
        self.calls = 0

    def find_collection(self, text: str) -> tuple[str, str] | None:
        self.calls += 1
        return self._found


def test_sets_collection_id_when_company_resolves() -> None:
    out = resolve_entity(
        Query(text="Mock Corp revenue 2023"), _Store(("us-cik-001234", "Mock Corp"))
    )
    assert out.filters["collection_id"] == "us-cik-001234"


def test_preserves_explicit_collection_filter() -> None:
    store = _Store(("us-cik-001234", "Mock Corp"))
    out = resolve_entity(
        Query(text="Mock Corp revenue", filters={"collection_id": "pinned"}), store
    )

    assert out.filters["collection_id"] == "pinned"
    assert store.calls == 0  # explicit filter wins; the store is not even queried


def test_no_match_leaves_query_unscoped() -> None:
    out = resolve_entity(Query(text="Globex revenue"), _Store(None))
    assert "collection_id" not in out.filters
