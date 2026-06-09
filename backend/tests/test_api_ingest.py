"""Tests for the document upload endpoint."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from finrag.api.deps import get_monitor_emitter, get_object_store


class _FakeStore:
    """Records puts in memory so the upload route stays hermetic (no MinIO)."""

    def __init__(self) -> None:
        self.puts: list[tuple[str, str, bytes, str]] = []

    async def put(
        self, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> None:
        self.puts.append((bucket, key, data, content_type))


class _FakeEmitter:
    """Records emitted monitor events so the upload announcement can be asserted."""

    def __init__(self) -> None:
        self.uploads: list[dict[str, object]] = []

    def upload(self, **event: object) -> None:
        self.uploads.append(event)

    def node(self, **event: object) -> None:  # pragma: no cover - endpoint never calls
        raise AssertionError("upload endpoint must not emit node events")

    def doc_done(self, **event: object) -> None:  # pragma: no cover - endpoint never calls
        raise AssertionError("upload endpoint must not emit doc_done events")


@pytest.fixture
def store() -> _FakeStore:
    return _FakeStore()


@pytest.fixture
def emitter() -> _FakeEmitter:
    return _FakeEmitter()


@pytest.fixture
def upload_client(
    client: TestClient, store: _FakeStore, emitter: _FakeEmitter
) -> Iterator[TestClient]:
    """The shared client with the object store + monitor emitter swapped for fakes."""
    client.app.dependency_overrides[get_object_store] = lambda: store
    client.app.dependency_overrides[get_monitor_emitter] = lambda: emitter
    yield client
    client.app.dependency_overrides.clear()


def _xbrl_bundle() -> list[tuple[str, tuple[str, bytes, str]]]:
    """A minimal complete US bundle: an inline-XBRL instance + its schema."""
    return [
        ("files", ("mock-10k/mock-20240928.htm", b"<xbrl/>", "text/html")),
        ("files", ("mock-10k/mock-20240928.xsd", b"<schema/>", "application/xml")),
        ("files", ("mock-10k/mock-20240928_lab.xml", b"<lab/>", "application/xml")),
    ]


def test_india_single_file_uploads_as_is(upload_client: TestClient, store: _FakeStore) -> None:
    response = upload_client.post(
        "/ingest/upload",
        data={"country": "india"},
        files=[("files", ("reliance-ar-fy25.pdf", b"%PDF-1.7", "application/pdf"))],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["country"] == "india"
    assert len(body["docs"]) == 1

    bucket, key, data, content_type = store.puts[0]
    assert bucket == "filings"
    assert key == f"india/{body['upload_id']}/reliance-ar-fy25.pdf"  # country + id + path
    assert data == b"%PDF-1.7"
    assert content_type == "application/pdf"
    assert body["docs"][0]["doc_id"] == key


def test_us_complete_bundle_uploads_all_files(upload_client: TestClient, store: _FakeStore) -> None:
    response = upload_client.post("/ingest/upload", data={"country": "us"}, files=_xbrl_bundle())
    assert response.status_code == 200
    body = response.json()
    assert len(body["docs"]) == 3
    # original folder + filenames preserved verbatim under the upload prefix
    keys = {put[1] for put in store.puts}
    assert keys == {
        f"us/{body['upload_id']}/mock-10k/mock-20240928.htm",
        f"us/{body['upload_id']}/mock-10k/mock-20240928.xsd",
        f"us/{body['upload_id']}/mock-10k/mock-20240928_lab.xml",
    }


def test_us_missing_schema_is_rejected(upload_client: TestClient, store: _FakeStore) -> None:
    response = upload_client.post(
        "/ingest/upload",
        data={"country": "us"},
        files=[("files", ("mock-10k/mock-20240928.htm", b"<xbrl/>", "text/html"))],
    )
    assert response.status_code == 422
    assert ".xsd" in response.json()["detail"]
    assert store.puts == []  # nothing written when validation fails


def test_us_missing_instance_is_rejected(upload_client: TestClient, store: _FakeStore) -> None:
    response = upload_client.post(
        "/ingest/upload",
        data={"country": "us"},
        files=[("files", ("mock-10k/mock-20240928.xsd", b"<schema/>", "application/xml"))],
    )
    assert response.status_code == 422
    assert store.puts == []


def test_unknown_country_is_rejected(upload_client: TestClient, store: _FakeStore) -> None:
    response = upload_client.post(
        "/ingest/upload",
        data={"country": "mars"},
        files=[("files", ("x.pdf", b"%PDF", "application/pdf"))],
    )
    assert response.status_code == 422  # enum validation
    assert store.puts == []


def test_upload_announces_the_batch_to_the_monitor(
    upload_client: TestClient, emitter: _FakeEmitter
) -> None:
    response = upload_client.post("/ingest/upload", data={"country": "us"}, files=_xbrl_bundle())
    body = response.json()

    assert len(emitter.uploads) == 1
    event = emitter.uploads[0]
    assert event["upload_id"] == body["upload_id"]
    assert event["country"] == "us"
    assert isinstance(event["created"], str) and event["created"]  # ISO timestamp stamped
    # one (doc_id, filename) pair per stored object, matching the response docs
    assert event["docs"] == [(doc["doc_id"], doc["filename"]) for doc in body["docs"]]


def test_rejected_upload_emits_nothing(upload_client: TestClient, emitter: _FakeEmitter) -> None:
    upload_client.post(
        "/ingest/upload",
        data={"country": "us"},
        files=[("files", ("mock-10k/mock-20240928.htm", b"<xbrl/>", "text/html"))],
    )
    assert emitter.uploads == []  # validation failed before any write/announce


def test_each_upload_gets_a_fresh_id(upload_client: TestClient, store: _FakeStore) -> None:
    first = upload_client.post(
        "/ingest/upload",
        data={"country": "india"},
        files=[("files", ("a.pdf", b"%PDF", "application/pdf"))],
    ).json()
    second = upload_client.post(
        "/ingest/upload",
        data={"country": "india"},
        files=[("files", ("a.pdf", b"%PDF", "application/pdf"))],
    ).json()
    assert first["upload_id"] != second["upload_id"]  # re-upload -> new id, pipeline dedups
