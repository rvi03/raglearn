"""Tests for the streaming segment screener.

Pins the three behaviours that make segment-screen correct: a clean stream is
forwarded as its *original* deltas (token-by-token UX preserved, just delayed to
the segment), the trailing segment is released on flush, and the first unsafe
segment is withheld with a single block signal and nothing after it leaks.
"""

from __future__ import annotations

from finrag.core.types import GuardVerdict
from finrag.security.output_stream import SegmentScreener


class _Guard:
    """Blocks any segment containing 'leak'."""

    def screen(self, text: str) -> GuardVerdict:
        if "leak" in text.lower():
            return GuardVerdict(allowed=False, category="system_prompt_leak", reason="x")
        return GuardVerdict(allowed=True)


def _screener(out: list[str], blocked: list[bool]) -> SegmentScreener:
    return SegmentScreener(
        guard=_Guard(), downstream=out.append, on_block=lambda: blocked.append(True)
    )


def test_clean_stream_forwards_original_deltas_after_flush() -> None:
    out: list[str] = []
    blocked: list[bool] = []
    screener = _screener(out, blocked)

    # No sentence boundary until flush, so nothing is released live.
    for delta in ["Net ", "sales ", "were ", "$1.2B [1]."]:
        screener(delta)
    assert out == []  # held until the segment closes

    screener.flush()
    assert out == ["Net ", "sales ", "were ", "$1.2B [1]."]  # same deltas, in order
    assert blocked == []


def test_completed_segment_releases_live() -> None:
    out: list[str] = []
    blocked: list[bool] = []
    screener = _screener(out, blocked)

    screener("First. ")  # sentence-ending punctuation + space → a boundary
    assert "".join(out) == "First. "  # released without waiting for flush
    screener("Second")
    assert "".join(out) == "First. "  # second sentence still buffered
    screener.flush()
    assert "".join(out) == "First. Second"


class _Redactor:
    """Masks the literal token 'jane@x.in' as an email."""

    def redact(self, text: str):
        from finrag.core.types import Redaction

        if "jane@x.in" in text:
            return Redaction(text=text.replace("jane@x.in", "[REDACTED:EMAIL]"), entities=["EMAIL"])
        return Redaction(text=text, entities=[])


def test_pii_in_a_segment_is_masked_before_forwarding() -> None:
    out: list[str] = []
    blocked: list[bool] = []
    screener = SegmentScreener(
        guard=_Guard(),
        downstream=out.append,
        on_block=lambda: blocked.append(True),
        redactor=_Redactor(),
    )

    for delta in ["Email ", "jane@x.in ", "for more."]:
        screener(delta)
    screener.flush()

    joined = "".join(out)
    assert "jane@x.in" not in joined  # the address never streamed
    assert "[REDACTED:EMAIL]" in joined
    assert blocked == []  # redaction is not a block


def test_clean_segment_with_redactor_still_replays_original_deltas() -> None:
    out: list[str] = []
    screener = SegmentScreener(
        guard=_Guard(), downstream=out.append, on_block=lambda: None, redactor=_Redactor()
    )
    for delta in ["No ", "pii ", "here."]:
        screener(delta)
    screener.flush()
    assert out == ["No ", "pii ", "here."]  # token granularity preserved when nothing is masked


def test_unsafe_segment_is_withheld_and_cuts_the_stream() -> None:
    out: list[str] = []
    blocked: list[bool] = []
    screener = _screener(out, blocked)

    screener("Safe start. ")  # clean segment → released
    screener("here is a leak. ")  # trips the guard → withheld
    screener("more after the leak")  # ignored once tripped

    assert "".join(out) == "Safe start. "  # only the clean prefix escaped
    assert "leak" not in "".join(out)
    assert blocked == [True]  # signalled exactly once
    assert screener.tripped is True

    screener.flush()  # flushing after a trip does nothing
    assert "".join(out) == "Safe start. "
    assert blocked == [True]
