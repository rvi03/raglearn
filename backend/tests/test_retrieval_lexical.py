"""Tests for the BM25 lexical retriever.

Pins what BM25 is for (exact-term matching dense can miss), and that it honours
the same scoping the dense path does: metadata filters and the secure-by-default
access-tag rule. A cold/empty index and a non-matching query are no-ops.
"""

from __future__ import annotations

from finrag.core.types import Chunk, ChunkType, DocumentMetadata, Market, Query
from finrag.retrieval.lexical import LexicalRetriever


def _chunk(
    chunk_id: str, text: str, *, collection: str = "c1", tags: list[str] | None = None
) -> Chunk:
    md = DocumentMetadata(
        collection_id=collection,
        company_name="Co",
        market=Market.US,
        filing_type="10-K",
        source_doc_id=chunk_id,
        access_tags=tags or [],
    )
    return Chunk(chunk_id=chunk_id, text=text, chunk_type=ChunkType.TEXT, metadata=md)


def test_ranks_exact_term_match_first() -> None:
    corpus = [
        _chunk("a", "general discussion of market conditions and outlook"),
        _chunk("b", "deferred revenue recognition policy under ASC 606"),
        _chunk("c", "risk factors and competition"),
    ]
    retriever = LexicalRetriever(corpus=corpus)

    results = retriever.retrieve(Query(text="deferred revenue recognition"), top_k=5)

    assert results[0].chunk.chunk_id == "b"  # the lexical match wins
    assert results[0].score > 0


def test_filters_scope_results_by_metadata() -> None:
    corpus = [
        _chunk("a", "revenue grew strongly", collection="c1"),
        _chunk("b", "revenue grew strongly", collection="c2"),
    ]
    retriever = LexicalRetriever(corpus=corpus)

    results = retriever.retrieve(Query(text="revenue", filters={"collection_id": "c2"}), top_k=5)

    assert [s.chunk.chunk_id for s in results] == ["b"]  # c1 filtered out


def test_access_tags_are_secure_by_default() -> None:
    corpus = [
        _chunk("public", "revenue figures", tags=[]),  # public
        _chunk("internal", "revenue figures", tags=["restricted"]),  # tagged
    ]
    retriever = LexicalRetriever(corpus=corpus)

    # Caller without the tag sees only the public chunk.
    public_only = retriever.retrieve(Query(text="revenue"), top_k=5)
    assert {s.chunk.chunk_id for s in public_only} == {"public"}

    # Caller holding the tag sees both.
    both = retriever.retrieve(Query(text="revenue", access_tags=["restricted"]), top_k=5)
    assert {s.chunk.chunk_id for s in both} == {"public", "internal"}


def test_empty_index_and_no_match_are_noops() -> None:
    assert LexicalRetriever(corpus=[]).retrieve(Query(text="anything"), top_k=5) == []
    retriever = LexicalRetriever(corpus=[_chunk("a", "revenue and earnings")])
    assert retriever.retrieve(Query(text="zzzznotpresent"), top_k=5) == []


def test_registered() -> None:
    from finrag.core.registry import registry

    assert isinstance(registry.create("retriever", "lexical", corpus=[]), LexicalRetriever)
