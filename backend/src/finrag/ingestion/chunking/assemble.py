"""Assemble an ordered item stream into section structure.

Both parse lanes -- the SEC HTML parser and the Docling PDF parser -- classify a
document into the same kind of ordered stream: headings that open sections, and
text/table content that fills them. This module turns that stream into a
:class:`~finrag.core.types.ParsedStructure`, so the two adapters share one
grouping rule and the chunker sees one shape.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from finrag.core.types import (
    BlockKind,
    ParsedStructure,
    StructureBlock,
    StructureSection,
)


class ParsedItem(NamedTuple):
    """A parsed element reduced to the role, text, depth, and page we keep.

    ``role`` is one of ``"title"`` (a top-level section heading), ``"subtitle"``
    (a nested heading), ``"text"`` (prose), or ``"table"`` (a rendered table).
    ``page`` is the source page when the parser provides one (Docling does; the
    SEC HTML parser does not).
    """

    role: str
    text: str
    level: int
    page: int | None = None


def assemble_sections(items: Sequence[ParsedItem], source_doc_id: str) -> ParsedStructure:
    """Group an ordered item stream into sections by their heading boundaries.

    A title or subtitle closes the open section and starts a new one; text and
    tables accumulate as blocks of the open section. Content before the first
    heading becomes an untitled preamble section. Empty headings collapse to an
    untitled section and blank blocks are dropped.

    Args:
      items: The classified elements in reading order.
      source_doc_id: The document the structure came from.

    Returns:
      The assembled sections in document order.
    """
    sections: list[StructureSection] = []
    current = StructureSection(title=None, level=0, blocks=[])
    for role, text, level, page in items:
        if role in ("title", "subtitle"):
            if current.blocks or current.title is not None:
                sections.append(current)
            current = StructureSection(title=text.strip() or None, level=level, blocks=[])
            continue
        body = text.strip()
        if body:
            kind = BlockKind.TABLE if role == "table" else BlockKind.TEXT
            current.blocks.append(StructureBlock(kind=kind, text=body, page=page))
    if current.blocks or current.title is not None:
        sections.append(current)
    return ParsedStructure(source_doc_id=source_doc_id, sections=sections)
