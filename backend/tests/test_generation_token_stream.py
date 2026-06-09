"""Tests for the live token-streaming seam."""

from __future__ import annotations

from finrag.generation.token_stream import (
    emit_token,
    reset_token_sink,
    set_token_sink,
    suppressed,
)


def test_emit_reaches_the_installed_sink() -> None:
    seen: list[str] = []
    token = set_token_sink(seen.append)
    try:
        emit_token("a")
        emit_token("b")
        emit_token("")  # empty deltas are dropped
    finally:
        reset_token_sink(token)
    assert seen == ["a", "b"]


def test_emit_is_a_noop_without_a_sink() -> None:
    emit_token("nothing listens")  # must not raise


def test_suppressed_mutes_emission_then_restores() -> None:
    seen: list[str] = []
    token = set_token_sink(seen.append)
    try:
        emit_token("before")
        with suppressed():
            emit_token("internal judge output")  # must not leak
        emit_token("after")
    finally:
        reset_token_sink(token)
    assert seen == ["before", "after"]
