"""Golden-set types and loader for the evaluation harness.

A golden set is the fixed list of cases every config/stage is scored against. It
is loaded from JSON so it can be curated by hand and versioned. The harness grows
per stage; the fields here cover what the ingestion and (later) retrieval stages
need: the query, what should be retrieved, the reference answer, and the expected
content-derived identity for metadata-accuracy scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldenCase(BaseModel):
    """One evaluation case: a query and what a correct system should produce."""

    id: str
    query: str
    query_type: str = "narrative"
    answerable: bool = True  # False = the corpus does NOT contain the answer; abstaining is correct
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_values: list[str] = Field(default_factory=list)  # figures a correct answer must state
    reference_answer: str | None = None
    expected_identity: dict[str, str] = Field(default_factory=dict)


def load_golden(path: str | Path) -> list[GoldenCase]:
    """Load a golden set from a JSON file (a list of case objects)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GoldenCase.model_validate(item) for item in data]
