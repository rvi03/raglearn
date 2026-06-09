"""Streaming output screen: vet a live token stream segment-by-segment.

The whole-answer :class:`~finrag.core.interfaces.OutputGuard` is the authority
on a finished answer, but ``/chat`` streams tokens as they are generated — so an
unsafe answer would reach the user *before* that final screen ever runs. This
sink closes the gap without giving up the live feel: it holds tokens until a
sentence boundary, screens the completed segment with the same guard, and only
then releases it. A clean segment is forwarded **as its original token deltas**
(so the UI still renders token-by-token, just delayed to the segment); the first
segment that trips the guard is withheld entirely and the stream is cut.

This is the segment-screen tradeoff, made explicit: clean text already released
stays out, but the offending segment never leaves the server. The same guard
verdict is reached here and by the whole-answer screen, so the two never disagree.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from finrag.core.interfaces.crosscutting import OutputGuard, PiiRedactor

# A segment boundary: sentence-ending punctuation followed by whitespace, or a
# run of newlines. Decimal points (``$1.2B``) are not boundaries — there is no
# whitespace after them — so figures are never split mid-number.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


class SegmentScreener:
    """A token sink that screens the stream one sentence at a time.

    Installed in place of the raw token sink on the ``/chat`` path. Tokens are
    buffered until a segment completes, the segment is screened, and a clean
    segment's original deltas are forwarded downstream. On a block the sink trips:
    it withholds the segment, invokes ``on_block`` once, and ignores all further
    tokens.
    """

    def __init__(
        self,
        *,
        guard: OutputGuard,
        downstream: Callable[[str], None],
        on_block: Callable[[], None],
        redactor: PiiRedactor | None = None,
    ) -> None:
        """Compose the screener over a downstream sink and a block signal.

        Args:
          guard: Screens each completed segment (the same guard the whole-answer
            check uses, so verdicts are consistent).
          downstream: Receives the original token deltas of a cleared segment.
          on_block: Called exactly once, when a segment first trips the guard.
          redactor: Optional PII redactor; when a cleared segment contains PII its
            redacted text is forwarded as one piece, else the original deltas are
            replayed. The high-confidence identifiers it catches are token-local,
            so segment-level redaction is sound (the whole-answer pass remains the
            authority for the returned answer).
        """
        self._guard = guard
        self._downstream = downstream
        self._on_block = on_block
        self._redactor = redactor
        self._pending: list[str] = []  # original deltas held until their segment clears
        self._tripped = False

    @property
    def tripped(self) -> bool:
        """Whether the guard has blocked a segment (the stream is then cut)."""
        return self._tripped

    def __call__(self, delta: str) -> None:
        """Buffer a token delta and release any segment it completes."""
        if self._tripped or not delta:
            return
        self._pending.append(delta)
        self._drain(final=False)

    def flush(self) -> None:
        """Release the trailing buffered text — the last segment has no boundary.

        Called once after generation finishes; a final segment that never ended in
        sentence punctuation is screened and released (or withheld) here.
        """
        self._drain(final=True)

    def _drain(self, *, final: bool) -> None:
        """Screen and release each completed segment from the buffer."""
        while not self._tripped:
            text = "".join(self._pending)
            match = _BOUNDARY.search(text)
            if match is not None:
                cut = match.end()
            elif final and text:
                cut = len(text)  # no trailing boundary: the remainder is the last segment
            else:
                return
            segment = text[:cut]
            if not self._guard.screen(segment).allowed:
                self._tripped = True
                self._pending.clear()
                self._on_block()
                return
            if self._redactor is not None:
                redacted = self._redactor.redact(segment)
                if redacted.entities:  # PII present → forward the masked segment, drop the deltas
                    self._downstream(redacted.text)
                    self._drop(cut)
                    continue
            self._release(cut)

    def _release(self, cut: int) -> None:
        """Forward the first ``cut`` chars downstream as original deltas; keep the rest.

        A delta that straddles the cut is split: its head is forwarded now and its
        tail stays buffered as the start of the next segment.
        """
        released = 0
        while self._pending and released < cut:
            piece = self._pending[0]
            if released + len(piece) <= cut:
                self._downstream(piece)
                released += len(piece)
                self._pending.pop(0)
            else:
                split = cut - released
                self._downstream(piece[:split])
                self._pending[0] = piece[split:]
                released = cut

    def _drop(self, cut: int) -> None:
        """Discard the first ``cut`` chars from the buffer without forwarding them.

        Used after a segment is redacted: its masked text was already forwarded, so
        the original deltas it came from must be consumed (a straddling delta keeps
        its tail buffered as the start of the next segment).
        """
        dropped = 0
        while self._pending and dropped < cut:
            piece = self._pending[0]
            if dropped + len(piece) <= cut:
                dropped += len(piece)
                self._pending.pop(0)
            else:
                self._pending[0] = piece[cut - dropped :]
                dropped = cut
