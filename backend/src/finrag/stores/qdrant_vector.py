"""Qdrant vector store: the persistent home for embedded narrative chunks.

Each chunk is one point carrying both a dense and a sparse (lexical) vector — the
hybrid signal bge-m3 produces — under named vectors ``dense`` and ``sparse``. The
point id is a deterministic UUID of the chunk id (itself versioned by content
hash), so re-ingesting identical content overwrites in place (idempotent) while a
new version lands as new points and coexists. The whole chunk is stored in the
payload, so search reconstructs it without a second lookup.

Search is hybrid when the query carries a sparse vector (dense + sparse prefetch
fused with Reciprocal Rank Fusion), dense-only otherwise. A metadata filter
narrows by payload fields (collection, access tags, ...). Cross-architecture
fusion and "latest version" precedence live above this layer.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from finrag.core.registry import registry
from finrag.core.types import Chunk, EmbeddingVector, ScoredChunk

_DENSE = "dense"
_SPARSE = "sparse"
# bge-m3's dense width.
_DENSE_DIM = 1024
# Stable namespace so a chunk id always maps to the same point id.
_POINT_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def _point_id(chunk_id: str) -> str:
    """Map a chunk id to a deterministic point UUID (Qdrant ids are int/UUID)."""
    return str(uuid.uuid5(_POINT_NS, chunk_id))


def _sparse_vector(vector: EmbeddingVector) -> models.SparseVector | None:
    """Convert an embedding's sparse weights to a Qdrant sparse vector, if present."""
    if not vector.sparse:
        return None
    return models.SparseVector(
        indices=list(vector.sparse.keys()), values=list(vector.sparse.values())
    )


@registry.register("vector_store", "qdrant")
class QdrantVectorStore:
    """A :class:`~finrag.core.interfaces.VectorStore` backed by Qdrant (hybrid)."""

    def __init__(
        self,
        url: str | None = None,
        *,
        location: str | None = None,
        collection: str = "finrag_chunks",
        dense_dim: int = _DENSE_DIM,
    ) -> None:
        """Connect to Qdrant and ensure the hybrid collection exists.

        Args:
          url: Qdrant server URL (production). Mutually exclusive with ``location``.
          location: In-process location, e.g. ``":memory:"`` (tests).
          collection: Collection name to read and write.
          dense_dim: Dense vector width (bge-m3 is 1024).
        """
        if location is not None:
            self._client = QdrantClient(location=location)
        elif url is not None:
            self._client = QdrantClient(url=url)
        else:
            raise ValueError("QdrantVectorStore needs either url or location")
        self._collection = collection
        self._dim = dense_dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection with named dense + sparse vectors if it is absent."""
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            self._collection,
            vectors_config={
                _DENSE: models.VectorParams(size=self._dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={_SPARSE: models.SparseVectorParams()},
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[EmbeddingVector]) -> None:
        """Insert or update chunks with their hybrid embeddings.

        Args:
          chunks: The chunks to store.
          vectors: Their embeddings, aligned to ``chunks``.
        """
        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            named: dict[str, object] = {_DENSE: list(vector.dense)}
            sparse = _sparse_vector(vector)
            if sparse is not None:
                named[_SPARSE] = sparse
            points.append(
                models.PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector=named,
                    payload=chunk.model_dump(mode="json"),
                )
            )
        if points:
            self._client.upsert(self._collection, points=points)

    def search(
        self,
        vector: EmbeddingVector,
        *,
        top_k: int,
        filters: dict[str, str],
        access_tags: Sequence[str] = (),
    ) -> list[ScoredChunk]:
        """Return the top-k chunks for a query vector under a metadata filter.

        Hybrid (dense + sparse fused with RRF) when the query has a sparse
        component, dense-only otherwise. The result is always scoped by the
        access clause, so a caller never sees chunks they lack a tag for.
        """
        query_filter = self._search_filter(filters, access_tags)
        dense = list(vector.dense)
        sparse = _sparse_vector(vector)
        if sparse is not None:
            response = self._client.query_points(
                self._collection,
                prefetch=[
                    models.Prefetch(query=dense, using=_DENSE, limit=top_k),
                    models.Prefetch(query=sparse, using=_SPARSE, limit=top_k),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        else:
            response = self._client.query_points(
                self._collection,
                query=dense,
                using=_DENSE,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        return [
            ScoredChunk(chunk=Chunk.model_validate(point.payload), score=point.score)
            for point in response.points
        ]

    def scroll(self) -> list[Chunk]:
        """Return every stored chunk (payload only, no vectors).

        Pages through the collection so the full corpus can seed an in-memory
        lexical index. Vectors are skipped — only the chunk payload is needed.
        """
        chunks: list[Chunk] = []
        offset: Any = None  # the Qdrant pagination cursor (opaque point id)
        while True:
            points, offset = self._client.scroll(
                self._collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(Chunk.model_validate(point.payload) for point in points)
            if offset is None:
                break
        return chunks

    def delete(self, filters: dict[str, str]) -> None:
        """Delete all points matching a metadata filter (GC of superseded versions).

        A no-op on empty filters — deleting everything is never the intent here.
        """
        query_filter = self._build_filter(filters)
        if query_filter is None:
            return
        self._client.delete(self._collection, points_selector=query_filter)

    @staticmethod
    def _eq_conditions(filters: dict[str, str]) -> list[models.FieldCondition]:
        """Equality conditions over payload metadata fields."""
        return [
            models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=value))
            for key, value in filters.items()
        ]

    @staticmethod
    def _access_clause(access_tags: Sequence[str]) -> models.Filter:
        """The visibility clause: public (untagged) chunks, plus any tagged for the caller.

        Modelled as a nested ``should`` (OR) so it can sit inside the search's
        ``must``: a chunk passes if its ``access_tags`` is empty *or* contains one
        of the caller's tags. With no caller tags, only public chunks pass.
        """
        key = "metadata.access_tags"
        shoulds: list[models.Condition] = [
            models.IsEmptyCondition(is_empty=models.PayloadField(key=key))
        ]
        if access_tags:
            shoulds.append(
                models.FieldCondition(key=key, match=models.MatchAny(any=list(access_tags)))
            )
        return models.Filter(should=shoulds)

    @classmethod
    def _search_filter(cls, filters: dict[str, str], access_tags: Sequence[str]) -> models.Filter:
        """Equality filters AND the access clause — the filter every search runs under."""
        return models.Filter(must=[*cls._eq_conditions(filters), cls._access_clause(access_tags)])

    @classmethod
    def _build_filter(cls, filters: dict[str, str]) -> models.Filter | None:
        """Build an equality-only filter (no access clause), for delete/GC.

        ``None`` on empty filters — deleting everything is never the intent, and
        the access clause must not leak into a delete selector.
        """
        if not filters:
            return None
        return models.Filter(must=list(cls._eq_conditions(filters)))
