"""Tests for the history-aware query rewrite (hermetic: fake LLM)."""

from __future__ import annotations

from finrag.core.interfaces.retrieval import QueryTransform
from finrag.core.registry import registry
from finrag.core.types import LLMResponse, Query, Usage
from finrag.retrieval.query_transform import RewriteTransform


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def generate(self, prompt: str) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self.reply, usage=Usage())


def test_no_history_is_a_noop_without_an_llm_call() -> None:
    llm = _FakeLLM("rewritten")
    out = RewriteTransform(llm).transform(Query(text="What about net income?"))

    assert out.text == "What about net income?"
    assert llm.calls == 0  # nothing to resolve, no call spent


def test_rewrites_a_follow_up_using_history() -> None:
    llm = _FakeLLM("What was Mock Corp's net income in 2023?")
    query = Query(text="What about net income?", history=["What was Mock Corp's revenue in 2023?"])

    out = RewriteTransform(llm).transform(query)

    assert out.text == "What was Mock Corp's net income in 2023?"
    assert out.history == query.history  # history preserved
    assert llm.calls == 1


def test_empty_rewrite_falls_back_to_original() -> None:
    out = RewriteTransform(_FakeLLM("   ")).transform(Query(text="orig", history=["h"]))
    assert out.text == "orig"


def test_registered_and_conforms_to_protocol() -> None:
    adapter = registry.create("query_transform", "rewrite", llm=_FakeLLM("x"))
    assert isinstance(adapter, RewriteTransform)
    assert isinstance(adapter, QueryTransform)
