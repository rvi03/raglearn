"""Fixed-size chunking: the structure-blind baseline and floor.

Flattens the whole document to text and splits it into fixed-size chunks with
overlap, ignoring section structure entirely. It exists for two reasons: it is
the baseline the structure-aware and semantic strategies must beat on the
evaluation leaderboard, and it is the guaranteed fallback when a document has no
recoverable structure at all.
"""

from __future__ import annotations

from collections.abc import Iterator

from raglearn.core.types import Chunk, ChunkType, DocumentMetadata, ParsedStructure
from raglearn.ingestion.chunking.recursive import recursive_split
from raglearn.ingestion.chunking.tokenizer import TokenCounter

_JOIN = "\n\n"


class FixedChunker:
    """Splits a document's flattened text into fixed-size, overlapping chunks."""

    def __init__(
        self,
        count_tokens: TokenCounter,
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
    ) -> None:
        """Configure the chunker with an embedder-aligned token counter and budget."""
        self._count = count_tokens
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(self, structure: ParsedStructure, metadata: DocumentMetadata) -> Iterator[Chunk]:
        """Yield fixed-size chunks over the document's flattened text.

        Section boundaries are not respected (that is the point of the
        baseline); chunks carry the filing's metadata with no section.
        """
        text = _JOIN.join(
            block.text for section in structure.sections for block in section.blocks
        )
        pieces = recursive_split(
            text,
            max_tokens=self._max_tokens,
            overlap_tokens=self._overlap_tokens,
            count_tokens=self._count,
        )
        for index, piece in enumerate(pieces):
            yield Chunk(
                chunk_id=f"{metadata.source_doc_id}-{index:04d}",
                text=piece,
                chunk_type=ChunkType.TEXT,
                metadata=metadata.model_copy(update={"section": None}),
            )
