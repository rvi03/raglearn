"""Pure identity logic shared by the market adapters.

No I/O and no library dependencies: hashing, the document-kind rules, the
kind-to-authority maps, and helpers for logical keys and period parsing. Keeping
these here (a) makes them unit-testable without Arelle or Docling and (b) keeps
the one hand-maintained vocabulary — the authority tiers — in a single place.
"""

from __future__ import annotations

import hashlib
import re

from finrag.core.types import NumericAuthority

# Canonical document kinds we normalize to. This is NOT a closed classifier
# vocabulary forced on the LLM/title — it is the small set of *behaviour-
# relevant* kinds we group and price trust by. Anything unrecognized is
# ``unknown`` and flagged for review.
ANNUAL_REPORT = "annual_report"
FINANCIAL_RESULTS = "financial_results"
INVESTOR_PRESENTATION = "investor_presentation"
PRESS_RELEASE = "press_release"
EARNINGS_CALL_TRANSCRIPT = "earnings_call_transcript"
UNKNOWN = "unknown"

# Title keywords → canonical kind, scanned in priority order (more specific
# first). Indian filings name their kind plainly on the cover page, so a keyword
# match over the Docling-extracted title classifies the common cases with no LLM.
_TITLE_RULES: tuple[tuple[str, str], ...] = (
    ("annual report", ANNUAL_REPORT),
    ("investor presentation", INVESTOR_PRESENTATION),
    ("earnings presentation", INVESTOR_PRESENTATION),
    # "transcript" specifically — bare "earnings/conference call" also appears in
    # press releases and decks announcing an upcoming call, so it is not a kind signal.
    ("transcript", EARNINGS_CALL_TRANSCRIPT),
    ("press release", PRESS_RELEASE),
    ("financial results", FINANCIAL_RESULTS),
    ("results for the quarter", FINANCIAL_RESULTS),
    ("results for the year", FINANCIAL_RESULTS),
    ("financial statements", FINANCIAL_RESULTS),
)

# Document kinds whose figures are audited/periodic and may be trusted as exact;
# everything else is issuer commentary (indicative).
_AUTHORITATIVE_DOCTYPES = frozenset({ANNUAL_REPORT, FINANCIAL_RESULTS})

# US SEC forms whose figures are audited/periodic. An amendment (``/A``) keeps
# its base form's authority. Forms not listed (8-K, 6-K, ...) are indicative.
_AUTHORITATIVE_FORMS = frozenset({"10-K", "10-Q", "20-F", "40-F", "11-K"})

# Period markers in a title, e.g. "Q4 FY26", "FY2024", "year ended March 31, 2026".
_PERIOD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bQ[1-4]\s*FY\s*\d{2,4}\b", re.IGNORECASE),
    re.compile(r"\bFY\s*\d{2,4}\b", re.IGNORECASE),
    re.compile(r"\b(?:quarter|year)\s+ended\s+[A-Za-z]+\s+\d{1,2},?\s*\d{4}\b", re.IGNORECASE),
)


def content_hash(data: bytes) -> str:
    """Return the SHA-256 hex digest of a document's bytes (its version id)."""
    return hashlib.sha256(data).hexdigest()


def slug(value: str | None) -> str:
    """Lowercase ``value`` and collapse non-alphanumerics to single underscores."""
    if not value:
        return "unknown"
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unknown"


def logical_key(company_id: str, doc_family: str, period: str | None) -> str:
    """Build the grouping key shared by every version of one logical document."""
    return f"{company_id}|{doc_family}|{period or ''}"


def form_family(form: str) -> str:
    """Strip a US amendment suffix so ``10-K/A`` groups with its ``10-K``."""
    return form.upper().removesuffix("/A")


def numeric_authority_for_form(form: str) -> NumericAuthority:
    """Map a US SEC form to its figure-trust tier (amendments keep base authority)."""
    if form_family(form) in _AUTHORITATIVE_FORMS:
        return NumericAuthority.AUTHORITATIVE
    return NumericAuthority.INDICATIVE


def numeric_authority_for_doctype(doc_type: str) -> NumericAuthority:
    """Map a canonical document kind to its figure-trust tier."""
    if doc_type in _AUTHORITATIVE_DOCTYPES:
        return NumericAuthority.AUTHORITATIVE
    return NumericAuthority.INDICATIVE


def doctype_from_title(title_text: str) -> str | None:
    """Return the canonical kind matched in a title, or ``None`` if none match."""
    haystack = title_text.lower()
    for keyword, kind in _TITLE_RULES:
        if keyword in haystack:
            return kind
    return None


def extract_period(text: str) -> str | None:
    """Return the first period marker found in ``text`` (normalized whitespace)."""
    for pattern in _PERIOD_PATTERNS:
        match = pattern.search(text)
        if match:
            return re.sub(r"\s+", " ", match.group(0)).strip()
    return None
