"""Tests for the token-aware recursive splitter.

Token counts are stood in by character length: deterministic, dependency-free,
and exact at the character rung, so the budget invariant can be asserted without
a real tokenizer.
"""

from __future__ import annotations

import pytest

from raglearn.ingestion.chunking.recursive import recursive_split


def char_tokens(text: str) -> int:
    """Stand-in token counter: one token per character."""
    return len(text)


def test_text_within_budget_returns_single_stripped_chunk() -> None:
    result = recursive_split("  Hello world.  ", max_tokens=100, count_tokens=char_tokens)

    assert result == ["Hello world."]


def test_blank_input_yields_no_chunks() -> None:
    assert recursive_split("   \n\n  ", max_tokens=10, count_tokens=char_tokens) == []
    assert recursive_split("", max_tokens=10, count_tokens=char_tokens) == []


def test_splits_on_coarsest_boundary_first() -> None:
    # Three paragraphs, each within budget: the split must cut on "\n\n" and
    # keep each paragraph whole rather than descend to finer separators.
    text = "aaaa\n\nbbbb\n\ncccc"

    result = recursive_split(text, max_tokens=5, count_tokens=char_tokens)

    assert result == ["aaaa", "bbbb", "cccc"]


def test_descends_to_sentence_when_line_overruns() -> None:
    # One line, no paragraph break, longer than budget: it must fall to the
    # sentence rung (". ").
    text = "First sentence. Second sentence. Third sentence."

    result = recursive_split(text, max_tokens=20, count_tokens=char_tokens)

    assert len(result) > 1
    assert all(char_tokens(chunk) <= 20 for chunk in result)


def test_unbreakable_run_falls_to_character_rung() -> None:
    # No whitespace or punctuation anywhere: only the character rung can bring
    # this within budget, and it must still terminate.
    text = "x" * 50

    result = recursive_split(text, max_tokens=5, count_tokens=char_tokens)

    assert len(result) > 1
    assert all(char_tokens(chunk) <= 5 for chunk in result)


def test_every_chunk_fits_budget_on_mixed_document() -> None:
    text = (
        "Heading paragraph with several words.\n\n"
        "A second paragraph that itself runs well past the budget and must be "
        "broken down further into sentence and word pieces. Another sentence here.\n\n"
        "Final."
    )

    result = recursive_split(text, max_tokens=30, count_tokens=char_tokens)

    assert result
    assert all(char_tokens(chunk) <= 30 for chunk in result)


def test_overlap_repeats_trailing_context_across_the_boundary() -> None:
    text = "alpha bravo charlie delta echo"

    result = recursive_split(text, max_tokens=13, overlap_tokens=6, count_tokens=char_tokens)

    # The word closing the first chunk is repeated at the start of the second,
    # so a fact straddling the cut keeps its context.
    assert len(result) >= 2
    assert "bravo" in result[0]
    assert "bravo" in result[1]


def test_no_overlap_when_overlap_tokens_is_zero() -> None:
    text = "alpha bravo charlie delta echo"

    result = recursive_split(text, max_tokens=11, overlap_tokens=0, count_tokens=char_tokens)

    # Concatenating the chunks reproduces each word exactly once -- nothing repeats.
    words = " ".join(result).split()
    assert words == ["alpha", "bravo", "charlie", "delta", "echo"]


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_rejects_non_positive_budget(max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens must be >= 1"):
        recursive_split("text", max_tokens=max_tokens, count_tokens=char_tokens)


@pytest.mark.parametrize("overlap_tokens", [10, 11, 20])
def test_rejects_overlap_not_smaller_than_budget(overlap_tokens: int) -> None:
    with pytest.raises(ValueError, match="overlap_tokens must be in"):
        recursive_split(
            "text", max_tokens=10, overlap_tokens=overlap_tokens, count_tokens=char_tokens
        )
