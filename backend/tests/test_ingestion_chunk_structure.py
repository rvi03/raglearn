"""Tests for the structure-aware chunker and context prefixing.

Token counts are stood in by character length so budgets are exact and
deterministic without loading a tokenizer.
"""

from __future__ import annotations

from finrag.core.types import (
    BlockKind,
    ChunkType,
    DocumentMetadata,
    Market,
    ParsedStructure,
    StructureBlock,
    StructureSection,
)
from finrag.ingestion.chunking.context import context_prefix, contextualize
from finrag.ingestion.chunking.structure import StructureChunker


def char_tokens(text: str) -> int:
    return len(text)


def _meta() -> DocumentMetadata:
    return DocumentMetadata(
        collection_id="c1",
        company_name="Mock Corp",
        market=Market.US,
        filing_type="10-K",
        fiscal_year=2024,
        source_doc_id="doc-1",
    )


def _section(title: str | None, *blocks: StructureBlock) -> StructureSection:
    return StructureSection(title=title, level=0, blocks=list(blocks))


def _text(t: str, page: int | None = None) -> StructureBlock:
    return StructureBlock(kind=BlockKind.TEXT, text=t, page=page)


def _table(t: str, page: int | None = None) -> StructureBlock:
    return StructureBlock(kind=BlockKind.TABLE, text=t, page=page)


def test_chunks_never_span_sections() -> None:
    structure = ParsedStructure(
        source_doc_id="doc-1",
        sections=[_section("Item 1", _text("aaa")), _section("Item 2", _text("bbb"))],
    )
    chunker = StructureChunker(char_tokens, max_tokens=100)

    chunks = list(chunker.chunk(structure, _meta()))

    assert [c.metadata.section for c in chunks] == ["Item 1", "Item 2"]
    assert [c.text for c in chunks] == ["aaa", "bbb"]


def test_consecutive_text_blocks_pack_up_to_the_budget() -> None:
    structure = ParsedStructure(
        source_doc_id="doc-1",
        sections=[_section("S", _text("aaaa"), _text("bbbb"), _text("cccc"))],
    )
    # budget 10: "aaaa\n\nbbbb" = 10 ok; adding cccc would exceed -> new chunk.
    chunker = StructureChunker(char_tokens, max_tokens=10)

    chunks = list(chunker.chunk(structure, _meta()))

    assert [c.text for c in chunks] == ["aaaa\n\nbbbb", "cccc"]
    assert all(c.chunk_type is ChunkType.TEXT for c in chunks)


def test_tables_are_kept_whole_and_flush_pending_text() -> None:
    structure = ParsedStructure(
        source_doc_id="doc-1",
        sections=[_section("S", _text("intro"), _table("| a | b |"), _text("after"))],
    )
    chunker = StructureChunker(char_tokens, max_tokens=100)

    chunks = list(chunker.chunk(structure, _meta()))

    assert [(c.text, c.chunk_type) for c in chunks] == [
        ("intro", ChunkType.TEXT),
        ("| a | b |", ChunkType.TABLE),
        ("after", ChunkType.TEXT),
    ]


def test_oversized_text_block_is_recursively_split_within_budget() -> None:
    big = "word " * 60  # 300 chars, no paragraph breaks
    structure = ParsedStructure(source_doc_id="doc-1", sections=[_section("S", _text(big))])
    chunker = StructureChunker(char_tokens, max_tokens=50, overlap_tokens=5)

    chunks = list(chunker.chunk(structure, _meta()))

    assert len(chunks) > 1
    assert all(char_tokens(c.text) <= 50 for c in chunks)


def test_page_and_identity_metadata_flow_onto_chunks() -> None:
    structure = ParsedStructure(
        source_doc_id="doc-1",
        sections=[_section("Item 7", _text("md&a", page=12))],
    )
    chunker = StructureChunker(char_tokens, max_tokens=100)

    chunk = next(iter(chunker.chunk(structure, _meta())))

    assert chunk.metadata.page == 12
    assert chunk.metadata.section == "Item 7"
    assert chunk.metadata.company_name == "Mock Corp"
    assert chunk.chunk_id == "doc-1-0000"


def test_chunk_ids_are_stable_and_ordered() -> None:
    structure = ParsedStructure(
        source_doc_id="doc-1",
        sections=[_section("S", _text("a"), _table("t"))],
    )
    chunker = StructureChunker(char_tokens, max_tokens=100)

    ids = [c.chunk_id for c in chunker.chunk(structure, _meta())]

    assert ids == ["doc-1-0000", "doc-1-0001"]


def test_context_prefix_builds_from_metadata_and_keeps_text_clean() -> None:
    structure = ParsedStructure(
        source_doc_id="doc-1",
        sections=[_section("Item 1A", _text("Risk prose."))],
    )
    chunk = next(iter(StructureChunker(char_tokens, max_tokens=100).chunk(structure, _meta())))

    assert context_prefix(chunk) == "Mock Corp · 10-K · 2024 · Item 1A"
    assert contextualize(chunk) == "Mock Corp · 10-K · 2024 · Item 1A\n\nRisk prose."
    assert chunk.text == "Risk prose."  # stored text stays clean
