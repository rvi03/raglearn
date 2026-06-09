"""Tests for the chat sessions CRUD endpoints (hermetic: in-memory fake store).

The store itself is integration-tested against real Postgres elsewhere; here we
override it with an in-memory fake to cover the route layer — response shapes and
the 404s on a missing conversation — without needing a database.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from finrag.api.deps import get_chat_store


class _FakeChatStore:
    """In-memory stand-in for :class:`PostgresChatStore` (only the parts routes use)."""

    def __init__(self) -> None:
        self._sessions: dict[str, str] = {}  # id -> title, insertion order = recency
        self._messages: dict[str, list[dict[str, Any]]] = {}

    def create_session(self, *, session_id: str, title: str, user_id: str = "local") -> None:
        self._sessions.setdefault(session_id, title)

    def list_sessions(self, *, user_id: str = "local", limit: int = 100) -> list[dict[str, Any]]:
        return [
            {"id": sid, "title": title, "created_at": "t", "updated_at": "t"}
            for sid, title in reversed(list(self._sessions.items()))
        ]

    def rename_session(self, *, session_id: str, title: str) -> bool:
        if session_id not in self._sessions:
            return False
        self._sessions[session_id] = title
        return True

    def delete_session(self, *, session_id: str) -> bool:
        existed = self._sessions.pop(session_id, None) is not None
        self._messages.pop(session_id, None)
        return existed

    def get_messages(self, *, session_id: str) -> list[dict[str, Any]]:
        return self._messages.get(session_id, [])


@pytest.fixture
def sessions_client(client: TestClient) -> Iterator[TestClient]:
    store = _FakeChatStore()
    client.app.dependency_overrides[get_chat_store] = lambda: store
    yield client
    client.app.dependency_overrides.clear()


def test_create_then_list(sessions_client: TestClient) -> None:
    created = sessions_client.post("/chat/sessions", json={"title": "Apple revenue"}).json()
    assert created["title"] == "Apple revenue"
    assert created["id"]

    sessions = sessions_client.get("/chat/sessions").json()["sessions"]
    assert [s["id"] for s in sessions] == [created["id"]]


def test_create_defaults_to_untitled(sessions_client: TestClient) -> None:
    created = sessions_client.post("/chat/sessions", json={}).json()
    assert created["title"] == "New analysis"


def test_rename_existing_and_missing(sessions_client: TestClient) -> None:
    sid = sessions_client.post("/chat/sessions", json={}).json()["id"]
    assert (
        sessions_client.patch(f"/chat/sessions/{sid}", json={"title": "renamed"}).status_code == 200
    )
    assert sessions_client.get("/chat/sessions").json()["sessions"][0]["title"] == "renamed"

    missing = sessions_client.patch("/chat/sessions/nope", json={"title": "x"})
    assert missing.status_code == 404


def test_delete_existing_and_missing(sessions_client: TestClient) -> None:
    sid = sessions_client.post("/chat/sessions", json={}).json()["id"]
    assert sessions_client.delete(f"/chat/sessions/{sid}").status_code == 200
    assert sessions_client.get("/chat/sessions").json()["sessions"] == []
    assert sessions_client.delete("/chat/sessions/nope").status_code == 404


def test_get_messages_empty_for_new_session(sessions_client: TestClient) -> None:
    sid = sessions_client.post("/chat/sessions", json={}).json()["id"]
    assert sessions_client.get(f"/chat/sessions/{sid}/messages").json() == {"messages": []}
