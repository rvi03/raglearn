"""Tests for the rules-based query router."""

from __future__ import annotations

from finrag.core.interfaces.retrieval import Router
from finrag.core.registry import registry
from finrag.core.types import Query
from finrag.retrieval.router import RulesRouter


def test_metric_query_routes_to_exact() -> None:
    router = RulesRouter()
    assert router.route(Query(text="What was Mock Corp's FY2023 net sales?")) == ["exact"]
    assert router.route(Query(text="Acme net income in 2024")) == ["exact"]


def test_non_metric_query_routes_to_narrative() -> None:
    router = RulesRouter()
    assert router.route(Query(text="What are the principal risks?")) == ["narrative"]
    assert router.route(Query(text="Summarize the outlook section")) == ["narrative"]


def test_registered_and_conforms_to_protocol() -> None:
    adapter = registry.create("router", "rules")
    assert isinstance(adapter, RulesRouter)
    assert isinstance(adapter, Router)
