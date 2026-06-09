"""Parse a PDF (or image) into section structure via Docling's vision pipeline.

Unlike SEC HTML, PDFs carry their structure visually, and Docling's vision
pipeline recovers it: a layout model labels each element (section header, text,
table) from font and position and records its page. This adapter walks that
converted document in reading order and normalizes it into the parser-agnostic
:class:`~finrag.core.types.ParsedStructure` the chunker consumes -- the same
shape the SEC HTML parser produces.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from io import BytesIO

from docling.datamodel.base_models import ConversionStatus, DocumentStream
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument, TableItem, TextItem
from docling_core.types.doc.labels import DocItemLabel

from finrag.core.errors import IngestionError
from finrag.core.types import ParsedStructure, RawDocument
from finrag.ingestion.chunking.assemble import ParsedItem, assemble_sections
from finrag.ingestion.docling_backend import build_converter

logger = logging.getLogger(__name__)

# Labels that open a section vs. carry body content. Page furniture (headers,
# footers, page numbers) and pictures carry no chunkable prose and are dropped.
_HEADING_LABELS = frozenset({DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE})
_BODY_LABELS = frozenset(
    {DocItemLabel.TEXT, DocItemLabel.PARAGRAPH, DocItemLabel.LIST_ITEM, DocItemLabel.CAPTION}
)


def _page_of(item: TextItem | TableItem) -> int | None:
    """Return the source page of an item, when Docling recorded provenance."""
    return item.prov[0].page_no if item.prov else None


def _to_items(doc: DoclingDocument) -> Iterator[ParsedItem]:
    """Yield the document's items in reading order as classified :class:`ParsedItem`s.

    Headings open sections (their depth taken from the document tree); text and
    list/caption items are prose; tables are rendered to markdown. Anything else
    is skipped.
    """
    for node, level in doc.iterate_items():
        if isinstance(node, TableItem):
            yield ParsedItem("table", node.export_to_markdown(doc), 0, _page_of(node))
        elif isinstance(node, TextItem):
            if node.label in _HEADING_LABELS:
                yield ParsedItem("title", node.text, level, _page_of(node))
            elif node.label in _BODY_LABELS:
                yield ParsedItem("text", node.text, 0, _page_of(node))


class DoclingStructureParser:
    """Parses a PDF/image document's bytes into :class:`ParsedStructure`."""

    def __init__(self, converter: DocumentConverter | None = None) -> None:
        """Bind to a converter, building the default one lazily on first use."""
        self._converter = converter

    @property
    def converter(self) -> DocumentConverter:
        """The Docling converter, built on first access to defer model loading."""
        if self._converter is None:
            self._converter = build_converter()
        return self._converter

    def parse(self, document: RawDocument) -> ParsedStructure:
        """Convert a document's bytes and normalize them to section structure.

        Args:
          document: The fetched document; ``data`` holds the bytes already
            retrieved from object storage.

        Returns:
          The document's sections and their content blocks.

        Raises:
          IngestionError: Docling could not convert the document.
        """
        stream = DocumentStream(
            name=document.filename or document.doc_id,
            stream=BytesIO(document.data or b""),
        )
        result = self.converter.convert(stream, raises_on_error=False)
        if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            raise IngestionError(f"docling failed to parse {document.doc_id}: {result.status.name}")
        if result.status is ConversionStatus.PARTIAL_SUCCESS:
            logger.warning("docling partially parsed %s", document.doc_id)
        return assemble_sections(list(_to_items(result.document)), document.doc_id)
