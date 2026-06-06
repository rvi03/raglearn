"""Parse a SEC HTML/inline-XBRL filing into section structure.

SEC filings style their headings with CSS on generic tags, so a DOM parser sees
a flat document. ``sec-parser`` instead reads the *visual* structure (font
weight/size, layout) and classifies each element by role -- top-section title
(``Part``/``Item``), sub-heading, paragraph, table -- which is exactly the
section structure a structure-aware chunker needs.

This adapter runs purely on the document bytes the ingestion consumer already
fetched from object storage; it never downloads anything. (``sec-parser`` ships
an EDGAR downloader as a transitive dependency, but it is never imported here.)
The library's element tree is normalized into the parser-agnostic
:class:`~raglearn.core.types.ParsedStructure` so the chunker stays independent of
``sec-parser``.
"""

from __future__ import annotations

import logging
from io import StringIO

import pandas as pd
import sec_parser as sp
from loguru import logger as _loguru_logger

from raglearn.core.types import ParsedStructure, RawDocument
from raglearn.ingestion.chunking.assemble import ParsedItem, assemble_sections

logger = logging.getLogger(__name__)

# sec-parser narrates its processing steps through loguru at warning level on
# some filings; silence its logger so it does not flood our logs. This affects
# only sec-parser's own logger namespace.
_loguru_logger.disable("sec_parser")

# One parser instance, reused across documents. The 10-Q parser handles 10-K and
# 20-F filings as well -- the element taxonomy (titles, text, tables) is shared.
_PARSER = sp.Edgar10QParser()


def _decode(data: bytes) -> str:
    """Decode filing bytes to text, tolerating non-UTF-8 legacy filings."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _table_to_markdown(element: sp.TableElement) -> str:
    """Render a table element to markdown, falling back to its flat text.

    The element's raw text concatenates cells with no delimiters, which embeds
    poorly. Parsing the table's own HTML into rows and emitting markdown keeps
    the row/column structure a retriever can use; a malformed table that pandas
    cannot read falls back to the flat text rather than failing the document.
    """
    try:
        frames = pd.read_html(StringIO(element.get_source_code()))
    except ValueError as error:
        logger.debug("table markdown fallback to text: %s", error)
        return element.text
    if not frames:
        return element.text
    return str(frames[0].to_markdown(index=False))


def _extract(element: sp.AbstractSemanticElement) -> ParsedItem | None:
    """Reduce a sec-parser element to a :class:`ParsedItem`, or ``None`` to skip.

    Titles and sub-headings open sections; text and tables are content. Page
    headers/footers, page numbers, images, empty nodes, and table-of-contents
    entries carry no chunkable content and are dropped. SEC HTML has no reliable
    page numbers, so ``page`` is left unset.
    """
    if isinstance(element, sp.TopSectionTitle):
        return ParsedItem("title", element.text, 0)
    if isinstance(element, sp.TitleElement):
        return ParsedItem("subtitle", element.text, max(1, getattr(element, "level", 0) + 1))
    if isinstance(element, (sp.TextElement, sp.SupplementaryText)):
        return ParsedItem("text", element.text, 0)
    if isinstance(element, sp.TableElement):
        return ParsedItem("table", _table_to_markdown(element), 0)
    return None


class SecHtmlParser:
    """Parses a SEC HTML/iXBRL document's bytes into :class:`ParsedStructure`."""

    def parse(self, document: RawDocument) -> ParsedStructure:
        """Parse a fetched SEC filing into its section structure.

        Args:
          document: The fetched filing; ``data`` holds the HTML bytes already
            retrieved from object storage.

        Returns:
          The filing's sections and their content blocks.
        """
        elements = _PARSER.parse(_decode(document.data or b""))
        items = [item for item in (_extract(element) for element in elements) if item is not None]
        return assemble_sections(items, document.doc_id)
