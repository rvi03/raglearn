"""Tests for the FIXED and SEMANTIC chunking strategies.

Hermetic: token counts are character length, and the semantic embedder is a
stand-in that maps text to fixed vectors, so a meaning boundary is forced
deterministically with no model.
"""

from __future__ import annotations

from collections.abc import Sequence

from raglearn.core.types import (
    BlockKind,
    ChunkType,
    DocumentMetadata,
    EmbeddingVector,
    Market,
    ParsedStructure,
    StructureBlock,
    StructureSection,
)
from raglearn.ingestion.chunking.fixed import FixedChunker
from raglearn.ingestion.chunking.semantic import SemanticChunker


def char_tokens(text: str) -> int:
    return len(text)


def _meta() -> DocumentMetadata:
    return DocumentMetadata(
        collection_id="c1",
        company_name="Mock Corp",
        market=Market.US,
        filing_type="10-K",
        source_doc_id="doc-1",
    )


def _structure(*texts: str) -> ParsedStructure:
    blocks = [StructureBlock(kind=BlockKind.TEXT, text=t) for t in texts]
    return ParsedStructure(
        source_doc_id="doc-1",
        sections=[StructureSection(title="S", level=0, blocks=blocks)],
    )


# ── FIXED ──────────────────────────────────────────────────────────────────


def test_fixed_ignores_structure_and_splits_flattened_text() -> None:
    structure = _structure("alpha", "bravo", "charlie")
    chunker = FixedChunker(char_tokens, max_tokens=12, overlap_tokens=0)

    chunks = list(chunker.chunk(structure, _meta()))

    assert all(c.metadata.section is None for c in chunks)
    assert all(c.chunk_type is ChunkType.TEXT for c in chunks)
    assert all(char_tokens(c.text) <= 12 for c in chunks)


def test_fixed_emits_no_chunks_for_empty_document() -> None:
    empty = ParsedStructure(source_doc_id="doc-1", sections=[])
    assert list(FixedChunker(char_tokens).chunk(empty, _meta())) == []


# ── SEMANTIC ───────────────────────────────────────────────────────────────


class _FakeEmbedder:
    """Embeds by keyword: 'risk' sentences map to one vector, others to another."""

    def embed(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        out = []
        for text in texts:
            dense = [1.0, 0.0] if "risk" in text.lower() else [0.0, 1.0]
            out.append(EmbeddingVector(dense=dense))
        return out


def test_semantic_cuts_at_the_meaning_boundary() -> None:
    # Three "risk" sentences then three "product" sentences -> one boundary.
    text = (
        "Risk one is real. Risk two is real. Risk three is real. "
        "Our product sells well. Our product is loved. Our product ships globally."
    )
    structure = _structure(text)
    chunker = SemanticChunker(
        _FakeEmbedder(),
        char_tokens,
        buffer_size=0,
        breakpoint_method="percentile",
        max_tokens=10_000,
    )

    chunks = list(chunker.chunk(structure, _meta()))

    assert len(chunks) == 2
    assert "risk" in chunks[0].text.lower()
    assert "product" in chunks[1].text.lower()
    assert all(c.metadata.section is None for c in chunks)


def test_semantic_size_caps_oversized_segments() -> None:
    text = "Risk one is real. Risk two is real. Risk three is also quite real here."
    structure = _structure(text)
    chunker = SemanticChunker(
        _FakeEmbedder(),
        char_tokens,
        buffer_size=0,
        breakpoint_method="percentile",
        max_tokens=20,
        overlap_tokens=4,
    )

    chunks = list(chunker.chunk(structure, _meta()))

    assert chunks
    assert all(char_tokens(c.text) <= 20 for c in chunks)


def test_semantic_single_sentence_is_one_chunk() -> None:
    structure = _structure("Only one sentence here.")
    chunker = SemanticChunker(_FakeEmbedder(), char_tokens, max_tokens=10_000)

    chunks = list(chunker.chunk(structure, _meta()))

    assert len(chunks) == 1
    assert chunks[0].text == "Only one sentence here."


def test_semantic_gradient_method_runs() -> None:
    text = "Risk one. Risk two. Product good. Product great. Product nice."
    chunker = SemanticChunker(_FakeEmbedder(), char_tokens, buffer_size=0, max_tokens=10_000)

    chunks = list(chunker.chunk(_structure(text), _meta()))

    assert chunks  # gradient method produces at least one chunk without error
