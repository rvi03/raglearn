"""Docling document parser for the PDF, IMAGE, and HTML lanes.

Docling's ``DocumentConverter`` is itself a router: it sends PDFs and raster
images through the vision pipeline (layout + OCR + TableFormer) and HTML through
the structural pipeline (digital text, no OCR), converging every format onto one
``DoclingDocument``. So a single :class:`DoclingParser` covers all three formats
the router hands it — the per-format pipeline split lives inside Docling, not in
our dispatch.

The vision pipeline is configured CPU-first (no torch-backed OCR, selective OCR,
accurate table structure); see :func:`_vision_pipeline_options`. XBRL is *not*
handled here — exact US figures come from Arelle on the facts path, not from
parsing the document into pages.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from io import BytesIO

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument
from rapidocr import LangDet, LangRec

from raglearn.core.errors import IngestionError
from raglearn.core.types import ParsedPage, RawDocument

logger = logging.getLogger(__name__)

# Formats this parser accepts. PDF and IMAGE take the vision pipeline; HTML takes
# the structural one. Anything else never reaches here — the router sends XML to
# the facts path and unknown bytes to quarantine.
_ACCEPTED_FORMATS = [InputFormat.PDF, InputFormat.IMAGE, InputFormat.HTML]


def _vision_pipeline_options() -> PdfPipelineOptions:
    """Build the locked CPU-first options for the PDF/IMAGE vision pipeline.

    The settings are deliberate, not defaults:

    - **Selective OCR** — ``do_ocr`` is on, but ``force_full_page_ocr`` is off
      and ``bitmap_area_threshold`` is 0.05, so Docling reads the digital text
      layer and only OCRs image regions covering more than 5% of a page. This
      recovers a table-rendered-as-image on an otherwise-digital page at no
      extra cost, and skips OCR entirely on clean digital pages.
    - **RapidOCR on ONNX Runtime** — CPU-only, no torch, permissively licensed.
      Its own default language is Chinese. Docling only honours ``lang`` when an
      ``artifacts_path`` of prefetched models is set; with on-demand downloads
      (no artifacts path) the model paths stay unset and RapidOCR falls back to
      its Chinese default. So English is forced at the engine level via
      ``rapidocr_params`` (``Det``/``Rec`` ``lang_type=en``), which selects the
      ``en_PP-OCRv4`` models regardless of the artifacts path.
    - **Accurate table structure with cell matching** — TableFormer reuses the
      PDF's real text cells instead of re-OCRing them, so figures stay exact.
    - **No page rasters** — image generation stays off until the multimodal
      vertical needs it.
    """
    ocr_options = RapidOcrOptions(
        backend="onnxruntime",
        lang=["english"],
        force_full_page_ocr=False,
        bitmap_area_threshold=0.05,
        rapidocr_params={"Det.lang_type": LangDet.EN, "Rec.lang_type": LangRec.EN},
    )
    return PdfPipelineOptions(
        do_ocr=True,
        ocr_options=ocr_options,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=True,
        ),
        generate_page_images=False,
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU,
            num_threads=os.cpu_count() or 1,
        ),
    )


def build_converter() -> DocumentConverter:
    """Build a converter wired with the locked vision options for PDF and IMAGE.

    HTML needs no custom options — Docling's default structural pipeline handles
    it — so only the vision formats carry an explicit option. PDF and IMAGE share
    the same locked pipeline options but bind to different backends (PDF parsing
    vs. image loading), so each gets its format-specific option object.
    """
    vision_options = _vision_pipeline_options()
    return DocumentConverter(
        allowed_formats=_ACCEPTED_FORMATS,
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=vision_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=vision_options),
        },
    )


def document_to_pages(doc: DoclingDocument) -> Iterator[ParsedPage]:
    """Map a converted document to pages, with tables kept separately.

    Paged formats (PDF, IMAGE) carry a real page layout, so each page is emitted
    with its own number and the tables whose provenance falls on it. HTML has no
    pages, so the whole document collapses to a single page. Each page's text is
    Docling's faithful markdown rendering (tables appear inline there too); the
    separate ``tables_markdown`` gives the structured path clean table access.

    Args:
      doc: The converted Docling document.

    Yields:
      One :class:`ParsedPage` per source page (or one page for HTML).
    """
    if doc.pages:
        for page_no in sorted(doc.pages):
            tables = [
                table.export_to_markdown(doc)
                for table in doc.tables
                if table.prov and table.prov[0].page_no == page_no
            ]
            yield ParsedPage(
                page_no=page_no,
                text=doc.export_to_markdown(page_no=page_no),
                tables_markdown=tables,
            )
    else:
        yield ParsedPage(
            page_no=1,
            text=doc.export_to_markdown(),
            tables_markdown=[table.export_to_markdown(doc) for table in doc.tables],
        )


class DoclingParser:
    """Parses PDF, IMAGE, and HTML documents into pages via Docling."""

    def __init__(self, converter: DocumentConverter) -> None:
        """Bind the parser to a configured converter.

        Args:
          converter: A Docling converter built by :func:`build_converter`.
        """
        self._converter = converter

    def parse(self, document: RawDocument) -> Iterator[ParsedPage]:
        """Convert a document's bytes and yield its pages.

        Args:
          document: The fetched document to parse.

        Yields:
          The document's parsed pages.

        Raises:
          IngestionError: Docling could not convert the document.
        """
        stream = DocumentStream(
            name=document.filename or document.doc_id,
            stream=BytesIO(document.data or b""),
        )
        result = self._converter.convert(stream, raises_on_error=False)
        if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            raise IngestionError(f"docling failed to parse {document.doc_id}: {result.status.name}")
        if result.status is ConversionStatus.PARTIAL_SUCCESS:
            logger.warning("docling partially parsed %s", document.doc_id)
        yield from document_to_pages(result.document)


def build_docling_parser() -> DoclingParser:
    """Build a :class:`DoclingParser` with the default configured converter."""
    return DoclingParser(build_converter())
