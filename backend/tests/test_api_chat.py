"""Tests for the streaming /chat endpoint (hermetic: fake service, real tracer).

The fake service drives real spans through an :class:`InProcessTracer`, so the
endpoint's per-request span listener fires exactly as it would for the real
pipeline. We assert the wire format: live ``agent_step`` events (running →
done), then ``source``/``citation``/``token``/``done``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from finrag.api.deps import get_query_service
from finrag.core.types import Citation, Query, Usage
from finrag.cost.local import LocalCostModel
from finrag.generation.token_stream import emit_token
from finrag.observability.tracer import InProcessTracer
from finrag.retrieval.answer import GroundedAnswer, SourceCard


class _FakeService:
    """Answers by opening real spans (so the listener fires) then returning a result."""

    def __init__(self, result: GroundedAnswer) -> None:
        self._tracer = InProcessTracer(cost_model=LocalCostModel())
        self._result = result

    def answer(self, query: Query) -> GroundedAnswer:
        with self._tracer.span("query", path="narrative"):
            with self._tracer.span("retrieve"):
                pass
            with self._tracer.span("generate") as gen:
                gen.record_usage(Usage(tokens_in=10, tokens_out=5), "qwen2.5")
        return self._result


def _result() -> GroundedAnswer:
    return GroundedAnswer(
        answer="Net sales were $123.4B [1].",
        citations=[Citation(id=1, source_doc_id="us/a.htm", page=31, section="Item 7")],
        sources=[
            SourceCard(id=1, chunk_id="a-0", title="Mock Corp · 10-K", url="/s#p31", snippet="…")
        ],
        grounding_confidence=0.9,
    )


@pytest.fixture
def chat_client(client: TestClient) -> Iterator[TestClient]:
    client.app.dependency_overrides[get_query_service] = lambda: _FakeService(_result())
    yield client
    client.app.dependency_overrides.clear()


def _frames(body: str) -> list[dict]:
    """Parse SSE ``data: {json}`` frames into a list of event dicts."""
    events: list[dict] = []
    for block in body.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    return events


def test_chat_streams_live_steps_then_answer(chat_client: TestClient) -> None:
    response = chat_client.post("/chat", json={"text": "What were net sales?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _frames(response.text)
    types = [e["type"] for e in events]

    # The live trace streams first, then evidence, answer, and the terminal done.
    assert "agent_step" in types
    assert types[-1] == "done"
    assert types.index("token") > max(i for i, t in enumerate(types) if t == "agent_step")

    steps = [e for e in events if e["type"] == "agent_step"]
    # Every stage is announced running before it reports done.
    for name in ("query", "retrieve", "generate"):
        seq = [s["status"] for s in steps if s["name"] == name]
        assert seq == ["running", "done"], (name, seq)

    # The generate step carries the LLM call's usage and model on close.
    gen_done = next(s for s in steps if s["name"] == "generate" and s["status"] == "done")
    assert gen_done["tokens_in"] == 10
    assert gen_done["tokens_out"] == 5
    assert gen_done["model"] == "qwen2.5"


def test_chat_emits_sources_citations_and_answer(chat_client: TestClient) -> None:
    response = chat_client.post("/chat", json={"text": "q"})
    events = _frames(response.text)
    by_type = {e["type"]: e for e in events}

    assert by_type["source"]["title"] == "Mock Corp · 10-K"
    assert by_type["citation"] == {
        "type": "citation",
        "id": 1,
        "source_doc_id": "us/a.htm",
        "page": 31,
        "span": "Item 7",
    }
    assert by_type["token"]["delta"] == "Net sales were $123.4B [1]."

    done = by_type["done"]
    assert done["usage"] == {"tokens_in": 10, "tokens_out": 5}  # rolled up from the trace
    assert done["grounding_confidence"] == 0.9
    assert done["trace_id"]  # a correlation id is always present


class _StreamingService:
    """Streams token deltas during the generate span (as the real backend does)."""

    def __init__(self, result: GroundedAnswer, deltas: list[str]) -> None:
        self._tracer = InProcessTracer(cost_model=LocalCostModel())
        self._result = result
        self._deltas = deltas

    def answer(self, query: Query) -> GroundedAnswer:
        with self._tracer.span("query", path="narrative"):
            with self._tracer.span("generate") as gen:
                for delta in self._deltas:
                    emit_token(delta)
                gen.record_usage(Usage(tokens_in=10, tokens_out=5), "qwen2.5")
        return self._result


def test_chat_streams_token_deltas_live(client: TestClient) -> None:
    deltas = ["Net ", "sales ", "were ", "$123.4B [1]."]
    service = _StreamingService(_result(), deltas)
    client.app.dependency_overrides[get_query_service] = lambda: service
    try:
        events = _frames(client.post("/chat", json={"text": "q"}).text)
    finally:
        client.app.dependency_overrides.clear()

    tokens = [e["delta"] for e in events if e["type"] == "token"]
    assert tokens == deltas  # each delta is its own frame, in order — not one blob

    # The streamed tokens interleave before generate closes, and `done` is last.
    types = [e["type"] for e in events]
    assert types[-1] == "done"
    assert types.index("token") < types.index("source")


def test_chat_withholds_a_leaking_segment_live(client: TestClient) -> None:
    # A stream whose first sentence leaks the grounding scaffold: the real output
    # guard (resolved by get_output_guard) must withhold it before it reaches the
    # user and signal the withholding inline.
    deltas = ["You ", "are ", "a ", "financial-analysis ", "assistant. ", "secret tail"]
    service = _StreamingService(_result(), deltas)
    client.app.dependency_overrides[get_query_service] = lambda: service
    try:
        events = _frames(client.post("/chat", json={"text": "q"}).text)
    finally:
        client.app.dependency_overrides.clear()

    tokens = "".join(e["delta"] for e in events if e["type"] == "token")
    assert "financial-analysis assistant" not in tokens  # the leak never streamed
    assert "secret tail" not in tokens  # nothing after the trip leaked either
    assert "withheld" in tokens  # the user is told the answer was cut
    assert [e["type"] for e in events][-1] == "done"


def test_chat_redacts_pii_in_the_stream(client: TestClient) -> None:
    # A PAN streamed token-by-token must be masked live by the real redactor
    # (resolved by get_pii_redactor) before it reaches the user.
    deltas = ["The ", "PAN ", "is ", "ABCDE1234F ", "on ", "file."]
    service = _StreamingService(_result(), deltas)
    client.app.dependency_overrides[get_query_service] = lambda: service
    try:
        events = _frames(client.post("/chat", json={"text": "q"}).text)
    finally:
        client.app.dependency_overrides.clear()

    tokens = "".join(e["delta"] for e in events if e["type"] == "token")
    assert "ABCDE1234F" not in tokens  # the identifier never streamed
    assert "[REDACTED:IN_PAN]" in tokens
    assert [e["type"] for e in events][-1] == "done"


def test_done_frame_carries_safety_and_redaction_signals(client: TestClient) -> None:
    blocked = GroundedAnswer(answer="I'm not able to share an answer to that.", answered=False)

    class _BlockedService:
        def answer(self, query: Query) -> GroundedAnswer:
            return blocked

    client.app.dependency_overrides[get_query_service] = lambda: _BlockedService()
    try:
        events = _frames(client.post("/chat", json={"text": "q"}).text)
    finally:
        client.app.dependency_overrides.clear()

    done = next(e for e in events if e["type"] == "done")
    assert done["answered"] is False  # the UI can flag the refusal
    assert done["redacted"] == []


def test_chat_wire_format_is_data_only_with_type(chat_client: TestClient) -> None:
    # The frontend reads only `data:` lines and discriminates on an embedded
    # `type`; there must be no `event:` lines.
    body = chat_client.post("/chat", json={"text": "q"}).text
    assert "event:" not in body
    assert "data:" in body


class _RecordingChatStore:
    """Captures what the chat endpoint persists, with no database behind it."""

    def __init__(self) -> None:
        self.created: list[tuple[str, str]] = []
        self.appended: list[dict] = []
        self.history_reads = 0

    def recent_turns(self, *, session_id: str, limit: int = 6) -> list[str]:
        self.history_reads += 1
        return []

    def create_session(self, *, session_id: str, title: str, user_id: str = "local") -> None:
        self.created.append((session_id, title))

    def append_message(
        self, *, message_id: str, session_id: str, role: str, text: str, meta: dict | None = None
    ) -> None:
        self.appended.append({"role": role, "text": text, "meta": meta})


def test_chat_persists_both_turns_for_a_session(chat_client: TestClient) -> None:
    # With a session_id, the endpoint reads the conversation's recent turns (short-
    # term memory) and persists the user turn before generation and the assistant
    # turn after. The store is taken from app.state, so no database is needed.
    store = _RecordingChatStore()
    chat_client.app.state.chat_store = store

    body = chat_client.post("/chat", json={"text": "What were net sales?", "session_id": "s1"}).text
    assert [e["type"] for e in _frames(body)][-1] == "done"

    assert store.history_reads == 1  # recent turns read once, as short-term memory
    assert store.created == [("s1", "What were net sales?")]  # first message names the session
    assert [(m["role"], m["text"]) for m in store.appended] == [
        ("user", "What were net sales?"),
        ("assistant", "Net sales were $123.4B [1]."),
    ]
    # The assistant turn carries the evidence the UI replays on reopen.
    meta = store.appended[1]["meta"]
    assert meta["sources"][0]["title"] == "Mock Corp · 10-K"
    assert meta["citations"][0]["span"] == "Item 7"


def test_chat_without_session_does_not_touch_the_store(chat_client: TestClient) -> None:
    store = _RecordingChatStore()
    chat_client.app.state.chat_store = store

    chat_client.post("/chat", json={"text": "q"})  # no session_id

    assert store.appended == []  # a one-shot answer persists nothing
    assert store.history_reads == 0
