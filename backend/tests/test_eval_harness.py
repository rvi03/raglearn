"""Tests for the evaluation harness skeleton: metrics, golden loader, leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

from finrag.eval import Leaderboard, load_golden, metrics

# --- metrics (pure math) ------------------------------------------------------


def test_recall_counts_expected_hits() -> None:
    assert metrics.recall(["a", "b", "c"], ["a", "c"]) == 1.0
    assert metrics.recall(["a"], ["a", "b"]) == 0.5
    assert metrics.recall([], ["a"]) == 0.0
    assert metrics.recall(["a"], []) == 1.0  # nothing expected -> trivially satisfied


def test_success_rate() -> None:
    assert metrics.success_rate([True, True, False, True]) == 0.75
    assert metrics.success_rate([]) == 1.0


def test_identity_accuracy_matches_expected_fields() -> None:
    predicted = {"market": "IN", "doc_type": "financial_results", "company": "mockpharma"}
    expected = {"market": "IN", "doc_type": "financial_results"}
    assert metrics.identity_accuracy(predicted, expected) == 1.0
    assert metrics.identity_accuracy({"market": "US"}, expected) == 0.0
    assert metrics.identity_accuracy({}, {}) == 1.0


def test_plumbing_leakage_flags_xbrl_artifacts() -> None:
    clean = ["Mock Corp's net sales rose.", "Risk factors include supply concentration."]
    assert metrics.plumbing_leakage(clean) == 0.0

    leaked = ["Revenue grew", "<link:schemaRef href='x.xsd'/>", "us-gaap:Revenues tagged here"]
    assert metrics.plumbing_leakage(leaked) == 2 / 3
    assert metrics.plumbing_leakage([]) == 0.0


def test_numeric_match_counts_expected_figures() -> None:
    answer = "Revenue was $4,250 million, up from $3,900 million."
    assert metrics.numeric_match(answer, ["4,250"]) == 1.0
    assert metrics.numeric_match(answer, ["4,250", "9,999"]) == 0.5
    assert metrics.numeric_match(answer, []) == 1.0  # nothing to check
    assert metrics.numeric_match("no figures here", ["4,250"]) == 0.0


def test_metric_registry_exposes_metrics_by_name() -> None:
    assert metrics.METRICS["recall"](["a"], ["a"]) == 1.0
    assert set(metrics.METRICS) >= {
        "recall",
        "success_rate",
        "identity_accuracy",
        "numeric_match",
        "plumbing_leakage",
    }


# --- golden loader ------------------------------------------------------------


def test_load_golden_parses_cases(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "query": "Mock Corp FY24 net income?",
                    "query_type": "exact",
                    "expected_chunk_ids": ["c1", "c2"],
                    "expected_identity": {"market": "US"},
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_golden(path)

    assert len(cases) == 1
    assert cases[0].id == "q1"
    assert cases[0].query_type == "exact"
    assert cases[0].expected_chunk_ids == ["c1", "c2"]
    assert cases[0].reference_answer is None  # defaulted


# --- leaderboard --------------------------------------------------------------


def test_leaderboard_ranks_by_metric() -> None:
    board = Leaderboard()
    board.add("config-a", {"recall": 0.7, "plumbing_leakage": 0.0})
    board.add("config-b", {"recall": 0.9, "plumbing_leakage": 0.1})
    board.add("config-c", {"recall": 0.5})

    ranked = board.ranked("recall")

    assert [row.config for row in ranked] == ["config-b", "config-a", "config-c"]
    assert board.ranked("recall", descending=False)[0].config == "config-c"
