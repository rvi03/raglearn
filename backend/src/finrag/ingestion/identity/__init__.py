"""Identity-extractor adapters and the per-country dispatcher.

The market is the one thing the path is trusted for: the enforced
``filings/<country>/`` prefix selects the adapter. Everything below the country
is free-form and identity comes from content.
"""

from __future__ import annotations

from finrag.core.errors import IngestionError
from finrag.core.interfaces.ingestion import IdentityExtractor
from finrag.core.registry import registry
from finrag.core.types import DocumentIdentity, DocumentMetadata
from finrag.ingestion.identity import india, us  # imports register the adapters

__all__ = ["india", "metadata_from_identity", "resolve_identity_extractor", "us"]

# Country prefix (first path segment of an object key) → registered adapter name.
_COUNTRY_TO_ADAPTER = {"us": "us", "india": "india"}


def resolve_identity_extractor(
    doc_id: str, *, classifier: india.Classifier | None = None
) -> IdentityExtractor:
    """Return the identity extractor for a document, chosen by its country prefix.

    Args:
      doc_id: The object key, e.g. ``us/appl/.../aapl-20230930.htm`` or
        ``india/natcopharma/Q4FY2026/results.pdf``. Its first segment is the
        country.
      classifier: Optional LLM doc-type classifier; injected into the India
        adapter as its rules fallback (ignored for other markets).

    Raises:
      IngestionError: The country prefix has no registered extractor.
    """
    country = doc_id.split("/", 1)[0].lower()
    name = _COUNTRY_TO_ADAPTER.get(country)
    if name is None:
        raise IngestionError(f"no identity extractor for country prefix: {country!r}")
    if name == "india" and classifier is not None:
        adapter = registry.create("identity_extractor", "india", classifier=classifier)
    else:
        adapter = registry.create("identity_extractor", name)
    result: IdentityExtractor = adapter
    return result


def metadata_from_identity(
    identity: DocumentIdentity,
    source_doc_id: str,
    *,
    section: str | None = None,
    page: int | None = None,
) -> DocumentMetadata:
    """Project a document's identity onto the per-chunk/fact metadata.

    Carries the versioned-identity fields (``content_hash``, ``logical_key``,
    ``numeric_authority``, ``recency``) onto every chunk so the stores can key on
    the version and resolve the latest at read time. ``section`` and ``page`` are
    set per chunk by the chunker.

    Args:
      identity: The document's content-derived identity.
      source_doc_id: The object key the chunk/fact is attributed to.
      section: The chunk's section title, if any.
      page: The chunk's source page, if any.
    """
    return DocumentMetadata(
        collection_id=identity.collection_id,
        company_name=identity.company or "unknown",
        ticker=identity.ticker,
        cik=identity.cik,
        market=identity.market,
        filing_type=identity.doc_type,
        fiscal_year=identity.fiscal_year,
        fiscal_period=identity.fiscal_period,
        section=section,
        source_doc_id=source_doc_id,
        page=page,
        content_hash=identity.content_hash,
        logical_key=identity.logical_key,
        numeric_authority=identity.numeric_authority,
        recency=identity.recency,
    )
