"""Tests for RRF fusion.

Pins the two properties that make RRF the right merge: a chunk ranked highly by
more than one retriever beats one ranked highly by a single retriever (agreement
is rewarded via summed contributions), and each chunk appears exactly once.
"""

from __future__ import annotations

from finrag.core.types import Chunk, ChunkType, DocumentMetadata, Market, ScoredChunk
from finrag.retrieval.fusion import RrfFusion


def _scored(chunk_id: str, score: float = 0.0) -> ScoredChunk:
    md = DocumentMetadata(
        collection_id="c1",
        company_name="Co",
        market=Market.US,
        filing_type="10-K",
        source_doc_id="us/x.htm",
    )
    chunk = Chunk(chunk_id=chunk_id, text="t", chunk_type=ChunkType.TEXT, metadata=md)
    return ScoredChunk(chunk=chunk, score=score)


def test_fuses_and_dedups_across_lists() -> None:
    dense = [_scored("a"), _scored("b"), _scored("c")]
    lexical = [_scored("c"), _scored("d")]

    fused = RrfFusion().fuse([dense, lexical])

    ids = [s.chunk.chunk_id for s in fused]
    assert sorted(ids) == ["a", "b", "c", "d"]  # union, each exactly once
    assert len(ids) == len(set(ids))  # deduped


def test_agreement_outranks_a_single_strong_list() -> None:
    # 'c' is 3rd in dense and 1st in lexical; 'a' is 1st in dense only. With k=60,
    # 'c' (1/61 + 1/63) should beat 'a' (1/61) — two retrievers agreeing wins.
    dense = [_scored("a"), _scored("b"), _scored("c")]
    lexical = [_scored("c"), _scored("e")]

    fused = RrfFusion().fuse([dense, lexical])

    assert fused[0].chunk.chunk_id == "c"
    assert fused[0].score > next(s.score for s in fused if s.chunk.chunk_id == "a")


def test_empty_input_returns_empty() -> None:
    assert RrfFusion().fuse([]) == []
    assert RrfFusion().fuse([[], []]) == []
