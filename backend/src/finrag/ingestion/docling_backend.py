"""Docling vision-pipeline configuration for the PDF/IMAGE structure parser.

Builds the Docling ``DocumentConverter`` used by the structure parser
(:mod:`finrag.ingestion.chunking.pdf_structure`). The converter routes PDFs
and raster images through Docling's vision pipeline (layout + OCR + TableFormer)
and HTML through its structural pipeline, converging on one ``DoclingDocument``.

The vision pipeline is configured CPU-first (no torch-backed OCR, selective OCR,
accurate table structure); see :func:`_vision_pipeline_options`. XBRL is not
handled here — exact US figures come from Arelle on the facts path.
"""

from __future__ import annotations

import os

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
from rapidocr import LangDet, LangRec

# Formats this converter accepts. PDF and IMAGE take the vision pipeline; HTML
# takes the structural one. XML never reaches here (it takes the facts path) and
# unknown bytes are quarantined upstream.
_ACCEPTED_FORMATS = [InputFormat.PDF, InputFormat.IMAGE, InputFormat.HTML]


def _vision_pipeline_options() -> PdfPipelineOptions:
    """Build the locked CPU-first options for the PDF/IMAGE vision pipeline.

    The settings are deliberate, not defaults:

    - **Selective OCR** — ``do_ocr`` is on, but ``force_full_page_ocr`` is off
      and ``bitmap_area_threshold`` is 0.05, so Docling reads the digital text
      layer and only OCRs image regions covering more than 5% of a page.
    - **RapidOCR on ONNX Runtime** — CPU-only, no torch, permissively licensed.
      Its own default language is Chinese, so English is forced at the engine
      level via ``rapidocr_params`` (``Det``/``Rec`` ``lang_type=en``).
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
