"""Evaluation harness.

Golden-set scoring, numeric and citation metrics, and the cross-config
comparison leaderboard live here.
"""

from finrag.eval import metrics
from finrag.eval.golden import GoldenCase, load_golden
from finrag.eval.leaderboard import Leaderboard, LeaderboardRow
from finrag.eval.runner import (
    CaseResult,
    evaluate_case,
    ingest_corpus,
    ingest_facts,
    run_eval,
    summarize,
)
from finrag.eval.synthetic import synthetic_corpus, synthetic_facts

__all__ = [
    "CaseResult",
    "GoldenCase",
    "Leaderboard",
    "LeaderboardRow",
    "evaluate_case",
    "ingest_corpus",
    "ingest_facts",
    "load_golden",
    "metrics",
    "run_eval",
    "summarize",
    "synthetic_corpus",
    "synthetic_facts",
]
