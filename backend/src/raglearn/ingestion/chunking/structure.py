"""Structure-aware chunking: pack a document's sections into sized chunks.

Walks the section structure in order and packs each section's content into
chunks up to the embedder's token budget, keeping section boundaries: a chunk
never spans two sections, so a retrieved chunk stays topically coherent and
carries one section's heading for citation. Tables are emitted whole as their
own chunks (splitting a table destroys it); a text run or a table that overruns
the budget is split with the token-aware recursive splitter as the floor.
"""

from __future__ import annotations

from collections.abc import Iterator

from raglearn.core.types import (
    BlockKind,
    Chunk,
    ChunkType,
    DocumentMetadata,
    ParsedStructure,
    StructureSection,
)
from raglearn.ingestion.chunking.recursive import recursive_split
from raglearn.ingestion.chunking.tokenizer import TokenCounter

_TEXT_JOIN = "\n\n"


class StructureChunker:
    """Packs a :class:`ParsedStructure` into section-coherent, sized chunks."""

    def __init__(
        self,
        count_tokens: TokenCounter,
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
    ) -> None:
        """Configure the chunker.

        Args:
          count_tokens: Token counter aligned to the embedder (e.g. bge-m3).
          max_tokens: Largest chunk size in tokens.
          overlap_tokens: Overlap carried when an oversized run is split.
        """
        self._count = count_tokens
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(self, structure: ParsedStructure, metadata: DocumentMetadata) -> Iterator[Chunk]:
        """Yield section-coherent chunks for a parsed document.

        Args:
          structure: The document's sections and blocks.
          metadata: The filing's binding metadata; each chunk inherits it with
            its own section and page filled in.

        Yields:
          Chunks in document order, each within the token budget, tables whole.
        """
        index = 0
        for section in structure.sections:
            for text, page, kind in self._section_units(section):
                yield Chunk(
                    chunk_id=f"{metadata.source_doc_id}-{index:04d}",
                    text=text,
                    chunk_type=kind,
                    metadata=metadata.model_copy(update={"section": section.title, "page": page}),
                )
                index += 1

    def _section_units(
        self, section: StructureSection
    ) -> Iterator[tuple[str, int | None, ChunkType]]:
        """Yield ``(text, page, type)`` chunk units for one section, in order.

        Consecutive text blocks pack together up to the budget; a table flushes
        any pending text then emits whole. Oversized text or tables fall to the
        recursive splitter.
        """
        pending: list[str] = []
        pending_page: int | None = None

        def flush() -> Iterator[tuple[str, int | None, ChunkType]]:
            nonlocal pending, pending_page
            if not pending:
                return
            joined = _TEXT_JOIN.join(pending)
            page = pending_page
            pending, pending_page = [], None
            yield from ((piece, page, ChunkType.TEXT) for piece in self._fit(joined))

        for block in section.blocks:
            if block.kind is BlockKind.TABLE:
                yield from flush()
                yield from ((piece, block.page, ChunkType.TABLE) for piece in self._fit(block.text))
                continue
            if pending and self._count(_TEXT_JOIN.join([*pending, block.text])) > self._max_tokens:
                yield from flush()
            if pending_page is None:
                pending_page = block.page
            pending.append(block.text)
        yield from flush()

    def _fit(self, text: str) -> list[str]:
        """Return ``text`` as-is if within budget, else recursively split it."""
        if self._count(text) <= self._max_tokens:
            return [text]
        return recursive_split(
            text,
            max_tokens=self._max_tokens,
            overlap_tokens=self._overlap_tokens,
            count_tokens=self._count,
        )
