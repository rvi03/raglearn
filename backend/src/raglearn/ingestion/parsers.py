"""Non-Docling parsers the router dispatches to.

The document-parsing lanes for PDF, IMAGE, and HTML live in
:mod:`raglearn.ingestion.docling_backend`; XBRL takes the facts arm in
:mod:`raglearn.ingestion.xbrl_extract`, not a page parser. This module holds the
rest of the page-parser table:

- :class:`TextParser` — plain text, decoded directly with no layout recovery.
- :class:`QuarantineParser` — the terminal route for unrecognized formats.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from raglearn.core.types import ParsedPage, RawDocument

logger = logging.getLogger(__name__)


def _decode_text(data: bytes) -> str:
    """Decode document bytes to text, tolerating legacy encodings.

    EDGAR's pre-inline-XBRL ``.txt`` filings are not always valid UTF-8, so a
    strict decode falls back to Latin-1 (which maps every byte and never
    raises) rather than dropping characters.
    """
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


class TextParser:
    r"""Plain text: decode the bytes and split into pages on form feeds.

    No layout recovery is needed — the content is already text. Legacy EDGAR
    ``.txt`` filings separate pages with the ASCII form feed (``\f``); splitting
    on it recovers real page numbers for citations. Blank pages (a trailing or
    repeated form feed) are dropped while their numbering is preserved, so a
    cited page number still points at the right page. Text with no form feed
    yields a single page.
    """

    def parse(self, document: RawDocument) -> Iterator[ParsedPage]:
        """Yield one :class:`ParsedPage` per form-feed-delimited page.

        Args:
          document: The fetched text document.

        Yields:
          A page for each non-blank form-feed segment, numbered from 1 in
          document order (blank segments consume a number but emit nothing).
        """
        text = _decode_text(document.data or b"")
        for page_no, body in enumerate(text.split("\f"), start=1):
            if body.strip():
                yield ParsedPage(page_no=page_no, text=body)


class QuarantineParser:
    """Terminal route for unrecognized formats: divert, never parse.

    A format the detector cannot place is held out of the pipeline rather than
    guessed at.
    """

    def parse(self, document: RawDocument) -> Iterator[ParsedPage]:
        """Record the quarantine and yield no pages.

        Args:
          document: The unrecognized document being diverted.

        Returns:
          An empty iterator; quarantined documents produce no pages.
        """
        logger.warning(
            "quarantined %s: unrecognized format (%d bytes)",
            document.doc_id,
            len(document.data or b""),
        )
        return iter(())
