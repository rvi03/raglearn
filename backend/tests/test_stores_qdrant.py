"""Tests for the Qdrant vector store (hermetic via in-memory Qdrant)."""

from __future__ import annotations

from finrag.core.registry import registry
from finrag.core.types import Chunk, ChunkType, DocumentMetadata, EmbeddingVector, Market
from finrag.stores.qdrant_vector import QdrantVectorStore


def _store() -> QdrantVectorStore:
    return QdrantVectorStore(location=":memory:", collection="t", dense_dim=4)


def _chunk(
    chunk_id: str, text: str, collection: str = "c1", tags: list[str] | None = None
) -> Chunk:
    md = DocumentMetadata(
        collection_id=collection,
        company_name="Co",
        market=Market.US,
        filing_type="10-K",
        source_doc_id="us/x.htm",
        content_hash="h",
        logical_key="lk",
        access_tags=tags or [],
    )
    return Chunk(chunk_id=chunk_id, text=text, chunk_type=ChunkType.TEXT, metadata=md)


def _vec(dense: list[float], sparse: dict[int, float] | None = None) -> EmbeddingVector:
    return EmbeddingVector(dense=dense, sparse=sparse)


def test_upsert_and_dense_search_roundtrip() -> None:
    store = _store()
    store.upsert(
        [_chunk("a-0", "risk factors"), _chunk("a-1", "revenue grew")],
        [_vec([1, 0, 0, 0]), _vec([0, 1, 0, 0])],
    )
    results = store.search(_vec([1, 0, 0, 0]), top_k=2, filters={})

    assert results[0].chunk.chunk_id == "a-0"
    assert results[0].chunk.text == "risk factors"  # full chunk rebuilt from payload


def test_hybrid_search_with_sparse() -> None:
    store = _store()
    store.upsert(
        [_chunk("a-0", "x"), _chunk("a-1", "y")],
        [_vec([1, 0, 0, 0], {1: 0.9}), _vec([0, 1, 0, 0], {2: 0.9})],
    )
    results = store.search(_vec([1, 0, 0, 0], {1: 0.9}), top_k=2, filters={})

    assert {r.chunk.chunk_id for r in results} == {"a-0", "a-1"}
    assert results[0].chunk.chunk_id == "a-0"  # matches both dense and sparse


def test_filter_restricts_by_metadata() -> None:
    store = _store()
    store.upsert(
        [_chunk("a-0", "x", collection="c1"), _chunk("b-0", "y", collection="c2")],
        [_vec([1, 0, 0, 0]), _vec([1, 0, 0, 0])],
    )
    results = store.search(_vec([1, 0, 0, 0]), top_k=5, filters={"collection_id": "c2"})

    assert [r.chunk.chunk_id for r in results] == ["b-0"]


def test_access_tags_isolate_restricted_chunks() -> None:
    store = _store()
    store.upsert(
        [_chunk("pub", "public", tags=[]), _chunk("sec", "restricted", tags=["finance"])],
        [_vec([1, 0, 0, 0]), _vec([1, 0, 0, 0])],
    )

    # No caller tags → only the public chunk is visible.
    anon = store.search(_vec([1, 0, 0, 0]), top_k=5, filters={})
    assert {r.chunk.chunk_id for r in anon} == {"pub"}

    # The matching tag unlocks the restricted chunk too.
    privileged = store.search(_vec([1, 0, 0, 0]), top_k=5, filters={}, access_tags=["finance"])
    assert {r.chunk.chunk_id for r in privileged} == {"pub", "sec"}

    # A non-matching tag still sees only public.
    other = store.search(_vec([1, 0, 0, 0]), top_k=5, filters={}, access_tags=["legal"])
    assert {r.chunk.chunk_id for r in other} == {"pub"}


def test_access_clause_combines_with_metadata_filter() -> None:
    store = _store()
    store.upsert(
        [
            _chunk("c1-pub", "x", collection="c1", tags=[]),
            _chunk("c2-sec", "y", collection="c2", tags=["finance"]),
        ],
        [_vec([1, 0, 0, 0]), _vec([1, 0, 0, 0])],
    )
    # Scoped to c2 AND privileged → the restricted c2 chunk.
    results = store.search(
        _vec([1, 0, 0, 0]), top_k=5, filters={"collection_id": "c2"}, access_tags=["finance"]
    )
    assert [r.chunk.chunk_id for r in results] == ["c2-sec"]


def test_idempotent_upsert_overwrites_same_chunk_id() -> None:
    store = _store()
    store.upsert([_chunk("a-0", "first")], [_vec([1, 0, 0, 0])])
    store.upsert([_chunk("a-0", "second")], [_vec([1, 0, 0, 0])])  # same id -> overwrite
    results = store.search(_vec([1, 0, 0, 0]), top_k=5, filters={})

    assert len(results) == 1
    assert results[0].chunk.text == "second"


def test_delete_by_filter_removes_matching_points() -> None:
    store = _store()
    store.upsert(
        [_chunk("a-0", "x", collection="c1"), _chunk("b-0", "y", collection="c2")],
        [_vec([1, 0, 0, 0]), _vec([1, 0, 0, 0])],
    )
    store.delete({"collection_id": "c1"})
    results = store.search(_vec([1, 0, 0, 0]), top_k=5, filters={})

    assert [r.chunk.chunk_id for r in results] == ["b-0"]


def test_delete_with_empty_filter_is_a_noop() -> None:
    store = _store()
    store.upsert([_chunk("a-0", "x")], [_vec([1, 0, 0, 0])])
    store.delete({})  # never "delete everything"
    assert len(store.search(_vec([1, 0, 0, 0]), top_k=5, filters={})) == 1


def test_registered() -> None:
    store = registry.create(
        "vector_store", "qdrant", location=":memory:", collection="r", dense_dim=4
    )
    assert isinstance(store, QdrantVectorStore)
