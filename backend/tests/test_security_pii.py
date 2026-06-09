"""Tests for the regex + checksum PII redactor.

Three directions matter: India and US identifiers are masked with the right type
label, financial prose (figures, percentages, fiscal years) is left untouched
(precision — the corpus is full of numbers), and the checksum-gated recognizers
(Aadhaar/Verhoeff, card/Luhn) reject structurally-valid-but-invalid numbers.
"""

from __future__ import annotations

import pytest

from finrag.core.registry import registry
from finrag.security.pii import RegexPiiRedactor

# A valid Aadhaar (passes Verhoeff) and a valid test card (passes Luhn).
_AADHAAR = "2341 2345 6783"  # Verhoeff-valid
_CARD = "4111 1111 1111 1111"  # Luhn-valid Visa test number


@pytest.mark.parametrize(
    ("text", "entity"),
    [
        ("Contact the director at jane.doe@natco.example.in for details.", "EMAIL"),
        ("The signatory's PAN is ABCDE1234F as on record.", "IN_PAN"),
        ("Registered under GSTIN 29ABCDE1234F1Z5 in Karnataka.", "IN_GSTIN"),
        ("Remit to IFSC HDFC0001234, account held in Mumbai.", "IN_IFSC"),
        ("Voter ID WBX1234567 was used as the address proof.", "IN_VOTER"),
        (f"Aadhaar number {_AADHAAR} was submitted with the KYC form.", "IN_AADHAAR"),
        ("Reach the promoter on +91 98765 43210 any weekday.", "IN_PHONE"),
        ("His SSN 123-45-6789 appears in the US filing.", "US_SSN"),
        ("Call investor relations at (415) 555-0132 for the transcript.", "US_PHONE"),
        (f"Payment card {_CARD} was on the expense report.", "CREDIT_CARD"),
    ],
)
def test_masks_identifiers_with_type_label(text: str, entity: str) -> None:
    result = RegexPiiRedactor().redact(text)
    assert entity in result.entities
    assert f"[REDACTED:{entity}]" in result.text


@pytest.mark.parametrize(
    "text",
    [
        "Mock Corp reported net revenue of $4,250,000,000 in FY2023 [1].",
        "Operating margin improved from 18.5% to 21.3% over three years.",
        "Total assets stood at 1,23,45,678 as per the Indian numbering format.",
        "The board met on 2024-09-30 to approve the results.",
        "EPS was 7000000000 basis units in the model (a bare figure, not a phone).",
    ],
)
def test_leaves_financial_prose_untouched(text: str) -> None:
    result = RegexPiiRedactor().redact(text)
    assert result.text == text  # nothing redacted
    assert result.entities == []


def test_checksum_rejects_invalid_aadhaar_and_card() -> None:
    # Right shape, wrong checksum → not treated as PII (no false positive).
    bad_aadhaar = RegexPiiRedactor().redact("ID 2341 2345 6788 is not a real Aadhaar.")
    assert bad_aadhaar.entities == []
    bad_card = RegexPiiRedactor().redact("Number 4111 1111 1111 1112 fails Luhn.")
    assert "CREDIT_CARD" not in bad_card.entities


def test_redacts_multiple_entities_in_one_pass() -> None:
    text = "Email jane@x.in, PAN ABCDE1234F, phone +91 98765 43210."
    result = RegexPiiRedactor().redact(text)
    assert set(result.entities) == {"EMAIL", "IN_PAN", "IN_PHONE"}
    assert "jane@x.in" not in result.text
    assert "ABCDE1234F" not in result.text


def test_registered() -> None:
    assert isinstance(registry.create("pii", "regex"), RegexPiiRedactor)
