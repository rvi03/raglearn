"""PDF cover-text signal for categorisation (not parsing).

Used only to *categorise* a document — never to build chunks. Docling's vision
pipeline labels graphic-heavy slides as pictures and drops them, so a deck's
title ("Earnings Presentation") is lost from the parsed structure even though it
is selectable vector text. ``pypdfium2`` reads that digital text layer directly,
and the PDF's embedded ``/Title`` metadata adds a second authoring-set signal.
Both feed the title rules; the chunk/parse path is untouched.

Best-effort: non-PDF bytes or any extraction error yield ``""`` so identity
falls back to the parsed structure (and then the LLM/review path).
"""

from __future__ import annotations

import logging

import pypdfium2 as pdfium

logger = logging.getLogger(__name__)


def pdf_cover_text(data: bytes) -> str:
    """Return a PDF's ``/Title`` metadata plus its first-page digital text.

    Args:
      data: The document bytes (may not be a PDF).

    Returns:
      The combined cover text for categorisation, or ``""`` if the bytes are not
      a readable PDF.
    """
    try:
        pdf = pdfium.PdfDocument(data)
    except Exception:  # not a PDF / unreadable - no signal, fall back
        return ""
    try:
        parts: list[str] = []
        title = pdf.get_metadata_dict().get("Title")
        if title:
            parts.append(str(title))
        if len(pdf) > 0:
            parts.append(pdf[0].get_textpage().get_text_range())
        return " \n ".join(part for part in parts if part)
    except Exception:
        logger.warning("pdf cover-text extraction failed; no categorisation signal", exc_info=True)
        return ""
    finally:
        pdf.close()
