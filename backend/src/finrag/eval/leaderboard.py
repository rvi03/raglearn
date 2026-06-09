"""Cross-config comparison leaderboard.

Collects one row of metric scores per config (or per architecture) and ranks
them by a chosen metric. This is the surface the comparison story (§7.9) is built
on: run the golden set across adapter configs, add a row each, rank.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LeaderboardRow(BaseModel):
    """One config's scores across the metrics it was evaluated on."""

    config: str
    scores: dict[str, float] = Field(default_factory=dict)


class Leaderboard(BaseModel):
    """An ordered set of config rows, rankable by any metric."""

    rows: list[LeaderboardRow] = Field(default_factory=list)

    def add(self, config: str, scores: dict[str, float]) -> None:
        """Append a config's metric scores."""
        self.rows.append(LeaderboardRow(config=config, scores=scores))

    def ranked(self, by: str, *, descending: bool = True) -> list[LeaderboardRow]:
        """Return the rows sorted by one metric (missing metric sorts as 0.0)."""
        return sorted(self.rows, key=lambda row: row.scores.get(by, 0.0), reverse=descending)
