"""Tests for the monitor emitters and upload-key parsing.

Pins the wire shape the monitor frontend reads (type-tagged JSON for upload/node/
doc_done), that detail is omitted when absent, and the upload-key recovery.
"""

from __future__ import annotations

import json

import pytest

from finrag.ingestion.monitor import split_upload_key
from finrag.observability.monitor_emitter import NullMonitorEmitter, RedisMonitorEmitter


class _FakePublisher:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    def publish(self, channel: str, message: str) -> int:
        self.published.append((channel, json.loads(message)))
        return 1


def test_upload_event_shape() -> None:
    pub = _FakePublisher()
    RedisMonitorEmitter(pub).upload(
        upload_id="u1",
        country="us",
        created="2026-06-08T00:00:00Z",
        docs=[("us/u1/a.htm", "a.htm"), ("us/u1/a.xsd", "a.xsd")],
    )
    channel, event = pub.published[0]
    assert channel == "finrag:ingestion"
    assert event["type"] == "upload"
    assert event["upload_id"] == "u1"
    assert event["country"] == "us"
    assert event["docs"] == [
        {"doc_id": "us/u1/a.htm", "filename": "a.htm"},
        {"doc_id": "us/u1/a.xsd", "filename": "a.xsd"},
    ]


def test_node_event_includes_detail_only_when_present() -> None:
    pub = _FakePublisher()
    emitter = RedisMonitorEmitter(pub)
    emitter.node(upload_id="u1", doc_id="d1", stage="parse", label="Parse", status="running")
    emitter.node(
        upload_id="u1", doc_id="d1", stage="chunk", label="Chunk", status="done", detail="42 chunks"
    )

    running = pub.published[0][1]
    assert running == {
        "type": "node",
        "upload_id": "u1",
        "doc_id": "d1",
        "id": "parse",
        "label": "Parse",
        "status": "running",
    }
    assert "detail" not in running  # omitted when None
    assert pub.published[1][1]["detail"] == "42 chunks"


def test_doc_done_event_shape() -> None:
    pub = _FakePublisher()
    RedisMonitorEmitter(pub).doc_done(upload_id="u1", doc_id="d1", outcome="indexed")
    assert pub.published[0][1] == {
        "type": "doc_done",
        "upload_id": "u1",
        "doc_id": "d1",
        "outcome": "indexed",
    }


def test_null_emitter_is_silent() -> None:
    emitter = NullMonitorEmitter()
    emitter.upload(upload_id="u1", country="us", created="t", docs=[])
    emitter.node(upload_id="u1", doc_id="d1", stage="parse", label="P", status="done")
    emitter.doc_done(upload_id="u1", doc_id="d1", outcome="indexed")  # must not raise


@pytest.mark.parametrize(
    ("doc_id", "expected"),
    [
        ("us/abc123/aapl-10k/aapl.htm", ("us", "abc123")),
        ("india/u9/results.pdf", ("india", "u9")),
        ("us/u1", None),  # no relpath segment → not a monitored upload
        ("loose.htm", None),
    ],
)
def test_split_upload_key(doc_id: str, expected: tuple[str, str] | None) -> None:
    assert split_upload_key(doc_id) == expected
