"""Tests for version-stable chunk ids."""

from __future__ import annotations

from typing import Any

from finrag.core.types import DocumentMetadata, Market
from finrag.ingestion.chunking.ids import make_chunk_id


def _md(**overrides: Any) -> DocumentMetadata:
    base: dict[str, Any] = {
        "collection_id": "c",
        "company_name": "Co",
        "market": Market.US,
        "filing_type": "10-K",
        "source_doc_id": "us/mockco/x.htm",
    }
    base.update(overrides)
    return DocumentMetadata(**base)


def test_uses_content_hash_when_present() -> None:
    md = _md(content_hash="abcdef0123456789ffffffff")
    assert make_chunk_id(md, 7) == "abcdef0123456789-0007"  # first 16 hex of the version


def test_falls_back_to_source_doc_id() -> None:
    assert make_chunk_id(_md(), 7) == "us/mockco/x.htm-0007"


def test_different_versions_get_different_ids() -> None:
    a = make_chunk_id(_md(content_hash="a" * 40), 0)
    b = make_chunk_id(_md(content_hash="b" * 40), 0)
    assert a != b
