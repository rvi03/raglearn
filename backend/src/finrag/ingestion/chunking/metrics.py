"""Intrinsic quality metrics for a chunking, used to pick the best per document.

Each metric scores a candidate chunking in ``[0, 1]`` (higher is better) without
any retrieval or ground truth, so the router can choose a strategy per document
cheaply. Two are cheap (size compliance, block integrity); two use the embedder
(intra-chunk cohesion, document coherence). A metric returns ``None`` when it
does not apply (e.g. coherence needs at least two chunks), and the gate then
drops it from the weighted average.

Block integrity is a containment proxy for the reference definition: a source
block is "intact" when its text survives whole inside a single chunk. It rewards
exactly what matters for filings -- keeping tables and clauses unsplit -- using
only the chunks and the source structure, since chunks carry no source offsets.
"""

from __future__ import annotations

import numpy as np

from finrag.core.interfaces.ingestion import Embedder
from finrag.core.types import Chunk, ParsedStructure
from finrag.ingestion.chunking.semantic import _split_sentences
from finrag.ingestion.chunking.tokenizer import TokenCounter


def _normalize(text: str) -> str:
    """Collapse whitespace so containment checks ignore formatting differences."""
    return " ".join(text.split())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, zero-safe."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def size_compliance(
    chunks: list[Chunk],
    count_tokens: TokenCounter,
    *,
    min_tokens: int = 100,
    max_tokens: int = 512,
) -> float | None:
    """Fraction of chunks whose token count is within ``[min_tokens, max_tokens]``."""
    if not chunks:
        return None
    out_of_band = sum(
        1 for chunk in chunks if not (min_tokens <= count_tokens(chunk.text) <= max_tokens)
    )
    return 1.0 - out_of_band / len(chunks)


def block_integrity(chunks: list[Chunk], structure: ParsedStructure) -> float | None:
    """Fraction of source blocks that survive whole inside a single chunk."""
    blocks = [block.text for section in structure.sections for block in section.blocks]
    if not blocks:
        return None
    normalized_chunks = [_normalize(chunk.text) for chunk in chunks]
    intact = sum(
        1
        for block in blocks
        if any(_normalize(block) in chunk_text for chunk_text in normalized_chunks)
    )
    return intact / len(blocks)


def intrachunk_cohesion(chunks: list[Chunk], embedder: Embedder) -> float | None:
    """Mean similarity of each chunk's sentences to the chunk as a whole."""
    scores: list[float] = []
    for chunk in chunks:
        sentences = _split_sentences(chunk.text)
        if len(sentences) < 2:
            continue
        vectors = embedder.embed([*sentences, chunk.text])
        chunk_vec = np.asarray(vectors[-1].dense, dtype=float)
        sims = [_cosine(np.asarray(v.dense, dtype=float), chunk_vec) for v in vectors[:-1]]
        scores.append(float(np.mean(sims)))
    if not scores:
        return None
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def document_coherence(
    chunks: list[Chunk],
    embedder: Embedder,
    count_tokens: TokenCounter,
    *,
    window_tokens: int = 3000,
) -> float | None:
    """Mean similarity of each chunk to the sliding window of chunks around it."""
    if len(chunks) < 2:
        return None
    embedded = embedder.embed([chunk.text for chunk in chunks])
    chunk_vectors = [np.asarray(v.dense, dtype=float) for v in embedded]
    sims: list[float] = []
    for start in range(len(chunks)):
        window: list[int] = []
        budget = 0
        index = start
        while index < len(chunks) and budget + count_tokens(chunks[index].text) <= window_tokens:
            budget += count_tokens(chunks[index].text)
            window.append(index)
            index += 1
        if len(window) < 2:
            continue
        window_text = " ".join(chunks[i].text for i in window)
        window_vec = np.asarray(embedder.embed([window_text])[0].dense, dtype=float)
        sims.extend(_cosine(chunk_vectors[i], window_vec) for i in window)
    if not sims:
        return None
    return float(np.clip(np.mean(sims), 0.0, 1.0))
