"""Pure evaluation metrics.

Stage-extensible: the ingestion-stage metrics live here now (parse success,
chunk recall, metadata/identity accuracy, plumbing leakage); retrieval and
generation metrics are added as those stages are built. Every metric is a pure
function in ``[0, 1]`` so they are trivially unit-tested and composed into a
leaderboard row.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

# Artifacts of the XBRL/bundle plumbing that must never leak into a narrative
# chunk; their presence means parsing/categorisation let raw markup through.
_PLUMBING_MARKERS = ("xbrli:", "<link:", "schemaref", "<xbrl", "linkbase", "us-gaap:")


def recall(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    """Fraction of the expected items present in the retrieved set (1.0 if none expected)."""
    expected_set = set(expected)
    if not expected_set:
        return 1.0
    return len(expected_set & set(retrieved)) / len(expected_set)


def success_rate(outcomes: Sequence[bool]) -> float:
    """Fraction of ``True`` outcomes — e.g. documents parsed without error (1.0 if empty)."""
    if not outcomes:
        return 1.0
    return sum(1 for outcome in outcomes if outcome) / len(outcomes)


def identity_accuracy(predicted: Mapping[str, str], expected: Mapping[str, str]) -> float:
    """Fraction of expected identity fields the prediction matches (1.0 if none expected)."""
    if not expected:
        return 1.0
    matched = sum(1 for key, value in expected.items() if predicted.get(key) == value)
    return matched / len(expected)


def numeric_match(answer_text: str, expected_values: Sequence[str]) -> float:
    """Fraction of expected values (figures/phrases) present verbatim in the answer.

    A proxy for numeric exact-match: a correct answer to "what was revenue?" must
    contain the expected figure. Returns 1.0 when nothing is expected (a
    narrative case with no figure to check).
    """
    if not expected_values:
        return 1.0
    found = sum(1 for value in expected_values if value in answer_text)
    return found / len(expected_values)


def plumbing_leakage(chunk_texts: Sequence[str]) -> float:
    """Fraction of chunks that leak XBRL/bundle plumbing into narrative text (target 0.0)."""
    if not chunk_texts:
        return 0.0
    leaked = sum(1 for text in chunk_texts if _has_plumbing(text))
    return leaked / len(chunk_texts)


def _has_plumbing(text: str) -> bool:
    """Return whether a chunk's text contains any plumbing marker."""
    lowered = text.lower()
    return any(marker in lowered for marker in _PLUMBING_MARKERS)


# Registry so the runner/leaderboard can look metrics up by name.
METRICS: dict[str, Callable[..., float]] = {
    "recall": recall,
    "success_rate": success_rate,
    "identity_accuracy": identity_accuracy,
    "numeric_match": numeric_match,
    "plumbing_leakage": plumbing_leakage,
}
