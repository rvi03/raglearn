"""Tests for the quality gate and the adaptive chunk router.

Hermetic: token counts are character length and the embedder is a constant
stand-in, so cohesion/coherence are equal across candidates and the cheap
metrics (size compliance, block integrity) drive the choice deterministically.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest

from raglearn.core.types import (
    BlockKind,
    Chunk,
    ChunkType,
    DocumentMetadata,
    EmbeddingVector,
    Market,
    ParsedStructure,
    StructureBlock,
    StructureSection,
)
from raglearn.ingestion.chunking.gate import QualityGate
from raglearn.ingestion.chunking.router import ChunkRouter


def char_tokens(text: str) -> int:
    return len(text)


class _ConstEmbedder:
    def embed(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        return [EmbeddingVector(dense=[1.0, 0.0]) for _ in texts]


def _meta() -> DocumentMetadata:
    return DocumentMetadata(
        collection_id="c1",
        company_name="Mock Corp",
        market=Market.US,
        filing_type="10-K",
        source_doc_id="doc-1",
    )


def _chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(chunk_id=f"doc-1-{index}", text=text, chunk_type=ChunkType.TEXT, metadata=_meta())


def _structure() -> ParsedStructure:
    blocks = [
        StructureBlock(kind=BlockKind.TEXT, text="aaaaa"),
        StructureBlock(kind=BlockKind.TEXT, text="bbbbb"),
    ]
    return ParsedStructure(
        source_doc_id="doc-1", sections=[StructureSection(title="S", level=0, blocks=blocks)]
    )


class _StubChunker:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def chunk(self, structure: ParsedStructure, metadata: DocumentMetadata) -> Iterator[Chunk]:
        return iter(self._chunks)


def _gate() -> QualityGate:
    return QualityGate(_ConstEmbedder(), char_tokens, min_tokens=1, max_tokens=20)


def test_gate_prefers_the_chunking_that_keeps_blocks_whole() -> None:
    good = [_chunk("aaaaa\n\nbbbbb")]  # both source blocks intact
    bad = [_chunk("aaa", 0), _chunk("aab", 1), _chunk("bbb", 2)]  # blocks split apart

    name, chunks, scores = _gate().select({"good": good, "bad": bad}, _structure())

    assert name == "good"
    assert scores["good"] > scores["bad"]
    assert chunks is good


def test_router_emits_the_winning_chunking() -> None:
    good = [_chunk("aaaaa\n\nbbbbb")]
    bad = [_chunk("aaa", 0), _chunk("aab", 1), _chunk("bbb", 2)]
    router = ChunkRouter({"good": _StubChunker(good), "bad": _StubChunker(bad)}, _gate())

    assert list(router.chunk(_structure(), _meta())) == good


def test_pinned_strategy_bypasses_the_gate() -> None:
    good = [_chunk("aaaaa\n\nbbbbb")]
    bad = [_chunk("aaa", 0)]
    router = ChunkRouter(
        {"good": _StubChunker(good), "bad": _StubChunker(bad)}, _gate(), pinned="bad"
    )

    assert list(router.chunk(_structure(), _meta())) == bad


def test_pinned_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="pinned strategy not registered"):
        ChunkRouter({"good": _StubChunker([])}, _gate(), pinned="nope")


def test_router_yields_nothing_when_all_strategies_are_empty() -> None:
    router = ChunkRouter({"a": _StubChunker([]), "b": _StubChunker([])}, _gate())

    assert list(router.chunk(_structure(), _meta())) == []
