"""Regex + checksum PII redactor (region-aware: India and US/global).

The privacy stage (OWASP LLM06). It masks personally identifiable information in
an answer before it is shown, without blocking the answer. Detection is purely
deterministic — regex structure plus a checksum where one exists — so it adds no
model weight to the image and never calls out. An NER-backed engine (Presidio)
can replace it behind the :class:`~finrag.core.interfaces.PiiRedactor` interface
when name/address detection is wanted.

Precision over recall, deliberately: the corpus is financial filings full of
numbers, so a redactor that ate figures would be worse than useless. Each
recognizer is therefore high-precision — distinctive letter/digit structures
(PAN, GSTIN, IFSC), separators, and checksums (Aadhaar = Verhoeff, card = Luhn) —
chosen so an ordinary monetary figure can't be mistaken for an identifier.
"""

from __future__ import annotations

import re

from finrag.core.registry import registry
from finrag.core.types import Redaction

# How a masked identifier is rendered in the output, e.g. ``[REDACTED:IN_PAN]``.
_MASK = "[REDACTED:{}]"


def _luhn_ok(digits: str) -> bool:
    """Return whether a digit string passes the Luhn checksum (payment cards)."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Verhoeff dihedral-group tables — the checksum Aadhaar numbers carry. Validating
# it turns a bare 12-digit match (which a large figure could be) into a
# high-confidence Aadhaar signal.
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def _verhoeff_ok(digits: str) -> bool:
    """Return whether a digit string passes the Verhoeff checksum (Aadhaar)."""
    check = 0
    for i, ch in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[i % 8][int(ch)]]
    return check == 0


def _luhn_validator(text: str) -> bool:
    return _luhn_ok(re.sub(r"\D", "", text))


def _aadhaar_validator(text: str) -> bool:
    return _verhoeff_ok(re.sub(r"\D", "", text))


# Each recognizer is (entity_type, pattern, optional validator). Patterns anchor on
# word boundaries; validators reject structurally-valid-but-uncertain matches.
_RECOGNIZERS: list[tuple[str, re.Pattern[str], object]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), None),
    # India — distinctive letter/digit shapes (very low false-positive on figures).
    ("IN_PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), None),
    ("IN_GSTIN", re.compile(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9][A-Z][0-9A-Z]\b"), None),
    ("IN_IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), None),
    ("IN_VOTER", re.compile(r"\b[A-Z]{3}[0-9]{7}\b"), None),
    # Aadhaar: 12 digits (first 2-9), optional 4-4-4 spacing, gated by Verhoeff.
    ("IN_AADHAAR", re.compile(r"\b[2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b"), _aadhaar_validator),
    # Indian mobile: require a +91 prefix or an internal separator, so a bare
    # 10-digit figure is never mistaken for a phone number.
    (
        "IN_PHONE",
        re.compile(r"\b(?:\+?91[\-\s]?[6-9][0-9]{9}|[6-9][0-9]{4}[\-\s][0-9]{5})\b"),
        None,
    ),
    # US/global — separators (SSN, phone) or a Luhn check (card) keep precision.
    ("US_SSN", re.compile(r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b"), None),
    (
        # No leading ``\b``: it would fail before a ``(`` in ``(415) …``. Bound on
        # word chars instead so the number isn't matched inside a longer token.
        "US_PHONE",
        re.compile(
            r"(?<!\w)(?:\+1[\-\s]?)?(?:\([0-9]{3}\)|[0-9]{3})[\-\s][0-9]{3}[\-\s][0-9]{4}(?!\w)"
        ),
        None,
    ),
    ("CREDIT_CARD", re.compile(r"\b[0-9](?:[\-\s]?[0-9]){12,18}\b"), _luhn_validator),
]


@registry.register("pii", "regex")
class RegexPiiRedactor:
    """A :class:`~finrag.core.interfaces.PiiRedactor` of regex + checksum rules."""

    def redact(self, text: str) -> Redaction:
        """Mask every detected identifier in ``text`` and report the types found.

        All recognizers run over the original text; matches are then applied left
        to right, skipping any that overlap an already-masked span, so the first
        (and longest-preferred) match for a region wins and nothing is double-cut.
        """
        spans: list[tuple[int, int, str]] = []
        for entity, pattern, validator in _RECOGNIZERS:
            for match in pattern.finditer(text):
                if validator is None or validator(match.group()):  # type: ignore[operator]
                    spans.append((match.start(), match.end(), entity))
        # Resolve overlaps: earliest start first, then the longer match.
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))

        out: list[str] = []
        found: set[str] = set()
        cursor = 0
        for start, end, entity in spans:
            if start < cursor:
                continue  # overlaps a span already masked
            out.append(text[cursor:start])
            out.append(_MASK.format(entity))
            found.add(entity)
            cursor = end
        out.append(text[cursor:])
        return Redaction(text="".join(out), entities=sorted(found))
