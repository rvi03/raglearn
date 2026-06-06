"""Tests for bucket-notification decoding."""

from __future__ import annotations

import json

import pytest

from raglearn.core.errors import IngestionError
from raglearn.ingestion.events import parse_object_events


def _put_event(key: str, *, content_type: str = "text/html", size: int = 42) -> str:
    """Build a minimal MinIO-style ObjectCreated notification for one object."""
    return json.dumps(
        {
            "EventName": "s3:ObjectCreated:Put",
            "Records": [
                {
                    "eventName": "s3:ObjectCreated:Put",
                    "s3": {
                        "bucket": {"name": "filings"},
                        "object": {
                            "key": key,
                            "size": size,
                            "contentType": content_type,
                            "eTag": "abc123",
                        },
                    },
                }
            ],
        }
    )


def test_decodes_single_put_event() -> None:
    events = parse_object_events(_put_event("apple%2F2025%2Fsub%2Fdeep.htm"))
    assert len(events) == 1
    event = events[0]
    assert event.bucket == "filings"
    assert event.key == "apple/2025/sub/deep.htm"  # %2F decoded to path separator
    assert event.event_name == "s3:ObjectCreated:Put"
    assert event.content_type == "text/html"
    assert event.size == 42
    assert event.etag == "abc123"


def test_decodes_space_encoded_as_plus() -> None:
    events = parse_object_events(_put_event("etsy%2FQ1+report.pdf"))
    assert events[0].key == "etsy/Q1 report.pdf"


def test_accepts_bytes_message() -> None:
    events = parse_object_events(_put_event("nvda%2F10k.htm").encode("utf-8"))
    assert events[0].key == "nvda/10k.htm"


def test_multiple_records_yield_multiple_events() -> None:
    payload = json.dumps(
        {
            "Records": [
                {
                    "eventName": "s3:ObjectCreated:Put",
                    "s3": {"bucket": {"name": "filings"}, "object": {"key": "a.htm"}},
                },
                {
                    "eventName": "s3:ObjectCreated:Put",
                    "s3": {"bucket": {"name": "filings"}, "object": {"key": "b.htm"}},
                },
            ]
        }
    )
    events = parse_object_events(payload)
    assert [e.key for e in events] == ["a.htm", "b.htm"]


def test_test_event_without_records_is_empty() -> None:
    assert parse_object_events(json.dumps({"EventName": "s3:TestEvent"})) == []


def test_invalid_json_raises() -> None:
    with pytest.raises(IngestionError, match="not valid JSON"):
        parse_object_events("{not json")


def test_unexpected_shape_raises() -> None:
    payload = json.dumps({"Records": [{"eventName": "x", "s3": {"bucket": {"name": "filings"}}}]})
    with pytest.raises(IngestionError, match="unexpected shape"):
        parse_object_events(payload)
