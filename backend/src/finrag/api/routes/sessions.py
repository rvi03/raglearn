"""Chat sessions endpoints: CRUD over named, persistent conversations.

The non-streaming companion to ``/chat``. The frontend rail uses these to list
past conversations (``GET``), start one (``POST``), rename/delete one
(``PATCH``/``DELETE``), and reopen its full transcript (``GET .../messages``).
``/chat`` itself appends the turns; this surface manages the conversations
around them.

Single-user for now: every session belongs to ``user_id='local'`` until per-user
authentication is added.
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from finrag.api.deps import get_chat_store
from finrag.stores.postgres_chat import PostgresChatStore

router = APIRouter(tags=["sessions"])

# Title shown for a conversation before its first message names it.
_UNTITLED = "New analysis"


class CreateSession(BaseModel):
    """Optional title for a new conversation (else ``New analysis``)."""

    title: str = _UNTITLED


class RenameSession(BaseModel):
    """A conversation's new title."""

    title: str


@router.post("/chat/sessions")
def create_session(
    body: CreateSession,
    store: Annotated[PostgresChatStore, Depends(get_chat_store)],
) -> dict[str, str]:
    """Create a conversation and return its id + title."""
    session_id = uuid4().hex
    store.create_session(session_id=session_id, title=body.title)
    return {"id": session_id, "title": body.title}


@router.get("/chat/sessions")
def list_sessions(
    store: Annotated[PostgresChatStore, Depends(get_chat_store)],
) -> dict[str, list[dict[str, Any]]]:
    """List conversations, most-recently-active first (the Recent rail)."""
    return {"sessions": store.list_sessions()}


@router.patch("/chat/sessions/{session_id}")
def rename_session(
    session_id: str,
    body: RenameSession,
    store: Annotated[PostgresChatStore, Depends(get_chat_store)],
) -> dict[str, str]:
    """Rename a conversation. 404 if it does not exist."""
    if not store.rename_session(session_id=session_id, title=body.title):
        raise HTTPException(status_code=404, detail="session not found")
    return {"id": session_id, "title": body.title}


@router.delete("/chat/sessions/{session_id}")
def delete_session(
    session_id: str,
    store: Annotated[PostgresChatStore, Depends(get_chat_store)],
) -> dict[str, str]:
    """Delete a conversation and its messages. 404 if it does not exist."""
    if not store.delete_session(session_id=session_id):
        raise HTTPException(status_code=404, detail="session not found")
    return {"id": session_id}


@router.get("/chat/sessions/{session_id}/messages")
def get_messages(
    session_id: str,
    store: Annotated[PostgresChatStore, Depends(get_chat_store)],
) -> dict[str, list[dict[str, Any]]]:
    """Return a conversation's full transcript in order (reopen a conversation)."""
    return {"messages": store.get_messages(session_id=session_id)}
