"""Semantic chunking: cut where meaning shifts, for documents without structure.

When a document has no usable headings, boundaries are found from meaning
instead: split into sentences, embed each within a small neighbour window, and
cut where the similarity between consecutive windows drops. This is the fallback
the router reaches for on flat documents; on well-structured filings the
structure strategy wins. The embedder is injected (the same bge-m3 used
downstream), so this has no model dependency of its own and is tested with a
stand-in embedder. Oversized segments fall to the recursive splitter.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import numpy as np

from raglearn.core.interfaces.ingestion import Embedder
from raglearn.core.types import Chunk, ChunkType, DocumentMetadata, ParsedStructure
from raglearn.ingestion.chunking.recursive import recursive_split
from raglearn.ingestion.chunking.tokenizer import TokenCounter

# Split on sentence-ending punctuation followed by whitespace and a capital or
# opening quote. The capital lookahead keeps decimals ("$1.2 million") and most
# mid-sentence abbreviations intact; it is a pragmatic splitter, not a parser.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_JOIN = "\n\n"


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, paragraph by paragraph."""
    sentences: list[str] = []
    for paragraph in text.split("\n"):
        stripped = paragraph.strip()
        if stripped:
            sentences.extend(part.strip() for part in _SENTENCE.split(stripped) if part.strip())
    return sentences


def _cosine_consecutive(vectors: np.ndarray) -> np.ndarray:
    """Return cosine similarity between each row and the next (length n-1)."""
    first, second = vectors[:-1], vectors[1:]
    norms = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    return np.asarray(np.einsum("ij,ij->i", first, second) / norms, dtype=float)


class SemanticChunker:
    """Cuts a document into chunks at semantic boundaries between sentences."""

    def __init__(
        self,
        embedder: Embedder,
        count_tokens: TokenCounter,
        *,
        buffer_size: int = 2,
        breakpoint_method: str = "gradient",
        breakpoint_percentile: float = 95.0,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
    ) -> None:
        """Configure the chunker.

        Args:
          embedder: Produces embeddings for sentence windows (e.g. bge-m3).
          count_tokens: Token counter aligned to the embedder, for the size cap.
          buffer_size: Neighbour sentences combined on each side before embedding.
          breakpoint_method: ``"gradient"`` (robust on homogeneous prose) or
            ``"percentile"`` (cut where the distance itself is high).
          breakpoint_percentile: Cut where the signal exceeds this percentile.
          max_tokens: Largest chunk size; oversized segments are recursively split.
          overlap_tokens: Overlap carried on an oversized split.
        """
        self._embedder = embedder
        self._count = count_tokens
        self._buffer_size = buffer_size
        self._method = breakpoint_method
        self._percentile = breakpoint_percentile
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(self, structure: ParsedStructure, metadata: DocumentMetadata) -> Iterator[Chunk]:
        """Yield semantically-bounded chunks over the document's flattened text."""
        text = _JOIN.join(block.text for section in structure.sections for block in section.blocks)
        segments = self._segment(text)
        for index, segment in enumerate(self._fit_all(segments)):
            yield Chunk(
                chunk_id=f"{metadata.source_doc_id}-{index:04d}",
                text=segment,
                chunk_type=ChunkType.TEXT,
                metadata=metadata.model_copy(update={"section": None}),
            )

    def _segment(self, text: str) -> list[str]:
        """Split text into semantic segments (joined sentence runs)."""
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return [text.strip()] if text.strip() else []

        windows = [self._window(sentences, i) for i in range(len(sentences))]
        vectors = np.array([vector.dense for vector in self._embedder.embed(windows)])
        distances = 1.0 - _cosine_consecutive(vectors)

        use_gradient = self._method == "gradient" and distances.size > 1
        signal = np.gradient(distances) if use_gradient else distances
        threshold = float(np.percentile(signal, self._percentile)) if signal.size else 0.0
        cut_after = [i for i, value in enumerate(signal) if value > threshold]

        segments: list[str] = []
        start = 0
        for boundary in cut_after:
            segments.append(" ".join(sentences[start : boundary + 1]))
            start = boundary + 1
        segments.append(" ".join(sentences[start:]))
        return [segment for segment in segments if segment.strip()]

    def _window(self, sentences: list[str], index: int) -> str:
        """Combine a sentence with its neighbours within the buffer."""
        low = max(0, index - self._buffer_size)
        high = min(len(sentences), index + self._buffer_size + 1)
        return " ".join(sentences[low:high])

    def _fit_all(self, segments: list[str]) -> Iterator[str]:
        """Yield segments, recursively splitting any that overrun the budget."""
        for segment in segments:
            if self._count(segment) <= self._max_tokens:
                yield segment
            else:
                yield from recursive_split(
                    segment,
                    max_tokens=self._max_tokens,
                    overlap_tokens=self._overlap_tokens,
                    count_tokens=self._count,
                )
