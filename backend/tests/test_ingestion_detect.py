"""Tests for Tika-backed format detection."""

from __future__ import annotations

import httpx
import pytest

from finrag.core.errors import IngestionError
from finrag.core.types import DetectedFormat
from finrag.ingestion.detect import TikaFormatDetector


def _detector_with_handler(handler: httpx.MockTransport) -> TikaFormatDetector:
    """Build a detector whose Tika calls are served by a mock transport."""
    detector = TikaFormatDetector(url="http://tika:9998")
    detector._client = httpx.Client(transport=handler)
    return detector


def _returning(media_type: str) -> TikaFormatDetector:
    """A detector whose Tika server always reports the given media type."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=media_type)

    return _detector_with_handler(httpx.MockTransport(handler))


@pytest.mark.parametrize(
    ("media_type", "expected"),
    [
        ("application/pdf", DetectedFormat.PDF),
        ("image/png", DetectedFormat.IMAGE),
        ("image/jpeg", DetectedFormat.IMAGE),
        ("image/tiff", DetectedFormat.IMAGE),
        ("image/bmp", DetectedFormat.IMAGE),
        ("image/webp", DetectedFormat.IMAGE),
        ("image/gif", DetectedFormat.IMAGE),
        ("text/html", DetectedFormat.HTML),
        ("application/xhtml+xml", DetectedFormat.HTML),
        ("application/xml", DetectedFormat.XML),
        ("text/xml", DetectedFormat.XML),
        ("text/plain", DetectedFormat.TEXT),
    ],
)
def test_detect_maps_known_media_types(media_type: str, expected: DetectedFormat) -> None:
    assert _returning(media_type).detect(b"some bytes") == expected


def test_detect_normalizes_media_type_parameters() -> None:
    assert _returning("text/html; charset=UTF-8").detect(b"<html>") == DetectedFormat.HTML


def test_detect_uppercase_media_type_is_normalized() -> None:
    assert _returning("Application/PDF").detect(b"%PDF-") == DetectedFormat.PDF


def test_detect_unrecognized_media_type_is_unknown() -> None:
    assert _returning("application/zip").detect(b"PK\x03\x04") == DetectedFormat.UNKNOWN


def test_detect_sends_only_the_byte_prefix() -> None:
    seen: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["length"] = len(request.content)
        return httpx.Response(200, text="application/pdf")

    detector = TikaFormatDetector(url="http://tika:9998", prefix_bytes=4)
    detector._client = httpx.Client(transport=httpx.MockTransport(handler))
    detector.detect(b"%PDF-1.7 followed by a much longer body that must be truncated")
    assert seen["length"] == 4


def test_detect_raises_ingestion_error_on_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    detector = _detector_with_handler(httpx.MockTransport(handler))
    with pytest.raises(IngestionError):
        detector.detect(b"data")


def test_detect_raises_ingestion_error_on_connection_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    detector = _detector_with_handler(httpx.MockTransport(handler))
    with pytest.raises(IngestionError):
        detector.detect(b"data")
