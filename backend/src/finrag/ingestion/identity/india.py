"""India identity from the parsed cover title.

Indian uploads have no XBRL, so identity comes from content: the document's
cover title (which Docling extracts during the parse the pages arm already runs)
is matched against keyword rules — deterministic, no LLM — to get the kind, and
a period marker is read from the title. An optional ``classifier`` is the LLM
fallback used only when the rules do not match; below a confidence gate the doc
is flagged for review and falls back to the upload folder's name as a weak
company hint. The path below ``filings/<country>/`` is a hint, never truth.
"""

from __future__ import annotations

from collections.abc import Callable

from finrag.core.errors import IngestionError
from finrag.core.registry import registry
from finrag.core.types import (
    DocumentIdentity,
    Market,
    ParsedStructure,
    RawDocument,
    XbrlExtraction,
)
from finrag.ingestion.identity import rules
from finrag.ingestion.identity.cover import pdf_cover_text

# An LLM fallback: given the title text, return (canonical_doc_type, confidence)
# or None if it cannot decide. Injected, optional; rules handle the common cases.
Classifier = Callable[[str], tuple[str, float] | None]

# Below this confidence the document's kind is not trusted: it is marked unknown
# and flagged for human tagging rather than silently mis-grouped.
_CONFIDENCE_GATE = 0.5


def _title_text(structure: ParsedStructure) -> str:
    """Gather the cover-page text to classify: headings and the lead text on page 1.

    The document's *kind* is declared on its cover, so only the cover page is
    scanned — restricting to it avoids a deep subheading (e.g. a press release's
    "Earnings Call Details") being mistaken for the document's kind. The cover is
    the smallest page number present; with no page info (HTML) only the first
    section is used.
    """
    pages = [b.page for s in structure.sections for b in s.blocks if b.page is not None]
    cover = min(pages) if pages else None
    parts: list[str] = []
    for section in structure.sections:
        on_cover = cover is None or any(b.page == cover for b in section.blocks)
        if section.title and on_cover:
            parts.append(section.title)
        for block in section.blocks:
            if block.text.strip() and (cover is None or block.page == cover):
                parts.append(block.text.strip()[:300])
                break
        if cover is None:
            break  # no page numbers (e.g. HTML): classify from the first section only
    return " \n ".join(parts)


def _path_hints(doc_id: str) -> tuple[str | None, str | None]:
    """Return (company_folder, period) hints from the object path, or ``None``s.

    Only a hint: ``filings/<country>/<org>/...`` puts the org folder right after
    the country, and a period marker may appear anywhere in the path.
    """
    parts = [p for p in doc_id.split("/") if p]
    company = parts[1] if len(parts) >= 2 else None
    period = rules.extract_period("/".join(parts[2:])) if len(parts) > 2 else None
    return company, period


def identity_from_structure(
    structure: ParsedStructure,
    document: RawDocument,
    *,
    classifier: Classifier | None = None,
) -> DocumentIdentity:
    """Derive an India document's identity from its parsed title (+ optional LLM)."""
    # Categorisation cover text: the PDF's digital text layer + /Title (which
    # carries graphic-slide titles Docling drops) plus the parsed structure title
    # (for HTML and as a fallback). This signal is used only to categorise.
    cover = pdf_cover_text(document.data or b"")
    title = f"{cover} \n {_title_text(structure)}".strip()
    matched = rules.doctype_from_title(title)
    confidence = 0.9
    if matched is None and classifier is not None:
        guess = classifier(title)
        if guess is not None:
            matched, confidence = guess
    if matched is not None and confidence >= _CONFIDENCE_GATE:
        doc_type = matched
        needs_review = False
    else:
        doc_type = rules.UNKNOWN
        confidence = min(confidence, 0.3)
        needs_review = True

    company_hint, path_period = _path_hints(document.doc_id)
    period = rules.extract_period(title) or path_period
    collection_id = f"in-{rules.slug(company_hint)}"
    return DocumentIdentity(
        market=Market.IN,
        collection_id=collection_id,
        logical_key=rules.logical_key(collection_id, doc_type, period),
        content_hash=rules.content_hash(document.data or b""),
        doc_type=doc_type,
        numeric_authority=rules.numeric_authority_for_doctype(doc_type),
        company=company_hint,
        fiscal_period=period,
        recency=None,
        confidence=confidence,
        needs_review=needs_review,
    )


@registry.register("identity_extractor", "india")
class IndiaIdentityExtractor:
    """An :class:`~finrag.core.interfaces.IdentityExtractor` for Indian uploads."""

    def __init__(self, *, classifier: Classifier | None = None) -> None:
        """Bind an optional LLM classifier used only when the title rules miss."""
        self._classifier = classifier

    def identify(
        self,
        document: RawDocument,
        *,
        structure: ParsedStructure | None = None,
        extraction: XbrlExtraction | None = None,
    ) -> DocumentIdentity:
        """Return the document's identity from its parsed cover title.

        Args:
          document: The triggering document (bytes give the content hash; the
            path gives the company/period hints).
          structure: The parsed structure from the pages arm. Required — the
            title to classify lives here.
          extraction: Unused for India (no XBRL).

        Raises:
          IngestionError: ``structure`` was not supplied.
        """
        if structure is None:
            raise IngestionError("india identity_extractor requires the parsed structure")
        return identity_from_structure(structure, document, classifier=self._classifier)
