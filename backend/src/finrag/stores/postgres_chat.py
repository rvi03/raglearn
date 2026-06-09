"""Postgres chat store: durable conversations (sessions + messages).

Turns the single, ephemeral chat into Claude-style **named, persistent
conversations**. A *session* is one conversation — auto-titled from its first
message, renamable, deletable; a *message* is one user or assistant turn within
it, kept with the metadata the UI needs to re-render it (citations, sources,
grounding, cost).

Two roles, one store:
- **writer** — the chat endpoint creates a session, then appends each user and
  assistant turn as it streams;
- **reader** — the sessions API lists conversations and replays a session's full
  history, and the chat endpoint reads the last few turns back as **short-term
  memory** (:meth:`recent_turns`) to feed the history-aware query rewrite.

Mirrors :class:`~finrag.stores.postgres_monitor.PostgresMonitorStore`: a small
``psycopg`` connection pool (FastAPI's sync-dependency threadpool reads from many
threads), schema ensured on construction. Single-user for now: every conversation
belongs to ``user_id='local'`` until per-user authentication is added.
"""

from __future__ import annotations

import json
from typing import Any

from psycopg_pool import ConnectionPool

# How many of a session's most recent turns feed the history-aware rewrite. Six
# messages ≈ three exchanges — enough to resolve a follow-up's coreference without
# bloating the rewrite prompt.
_RECENT_TURNS = 6

_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id         TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL DEFAULT 'local',
        title      TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT current_timestamp,
        updated_at TIMESTAMP DEFAULT current_timestamp
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions (user_id, updated_at DESC)",
    # ON DELETE CASCADE: deleting a session drops its whole transcript in one step.
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id         TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES chat_sessions (id) ON DELETE CASCADE,
        role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        text       TEXT NOT NULL,
        meta       JSONB,
        created_at TIMESTAMP DEFAULT current_timestamp
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chat_messages_session "
    "ON chat_messages (session_id, created_at)",
)


class PostgresChatStore:
    """Durable conversation store (sessions + messages) backed by Postgres."""

    def __init__(self, dsn: str) -> None:
        """Open a connection pool and ensure the schema exists.

        Args:
          dsn: Postgres connection string.
        """
        self._pool = ConnectionPool(dsn, min_size=1, open=True)
        with self._pool.connection() as conn:
            for statement in _SCHEMA:
                conn.execute(statement)

    # --- sessions -------------------------------------------------------------

    def create_session(self, *, session_id: str, title: str, user_id: str = "local") -> None:
        """Create a new conversation. Idempotent on ``session_id`` (no-op if it exists)."""
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO chat_sessions (id, user_id, title) VALUES (%s, %s, %s) "
                "ON CONFLICT (id) DO NOTHING",
                (session_id, user_id, title),
            )

    def list_sessions(self, *, user_id: str = "local", limit: int = 100) -> list[dict[str, Any]]:
        """Return a user's conversations, most-recently-active first."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM chat_sessions "
                "WHERE user_id = %s ORDER BY updated_at DESC LIMIT %s",
                (user_id, limit),
            ).fetchall()
        return [
            {
                "id": sid,
                "title": title,
                "created_at": created_at.isoformat(),
                "updated_at": updated_at.isoformat(),
            }
            for sid, title, created_at, updated_at in rows
        ]

    def rename_session(self, *, session_id: str, title: str) -> bool:
        """Set a conversation's title. Returns whether the session existed."""
        with self._pool.connection() as conn:
            cur = conn.execute(
                "UPDATE chat_sessions SET title = %s, updated_at = current_timestamp WHERE id = %s",
                (title, session_id),
            )
            return cur.rowcount > 0

    def delete_session(self, *, session_id: str) -> bool:
        """Delete a conversation and its messages (CASCADE). Returns whether it existed."""
        with self._pool.connection() as conn:
            cur = conn.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
            return cur.rowcount > 0

    # --- messages -------------------------------------------------------------

    def append_message(
        self,
        *,
        message_id: str,
        session_id: str,
        role: str,
        text: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Append one turn and bump its session's ``updated_at`` (recency for the rail)."""
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO chat_messages (id, session_id, role, text, meta) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    message_id,
                    session_id,
                    role,
                    text,
                    json.dumps(meta) if meta is not None else None,
                ),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = current_timestamp WHERE id = %s",
                (session_id,),
            )

    def get_messages(self, *, session_id: str) -> list[dict[str, Any]]:
        """Return a session's full transcript in order (for reopening a conversation)."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, role, text, meta, created_at FROM chat_messages "
                "WHERE session_id = %s ORDER BY created_at, id",
                (session_id,),
            ).fetchall()
        return [
            {
                "id": mid,
                "role": role,
                "text": text,
                "meta": meta,
                "created_at": created_at.isoformat(),
            }
            for mid, role, text, meta, created_at in rows
        ]

    def recent_turns(self, *, session_id: str, limit: int = _RECENT_TURNS) -> list[str]:
        """Return the last ``limit`` turns as labelled lines for the history rewrite.

        Each line is ``"User: …"`` / ``"Assistant: …"`` in chronological order — the
        shape :class:`~finrag.core.types.Query` ``history`` is joined into a
        ``CONVERSATION`` block for coreference resolution.
        """
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT role, text FROM chat_messages WHERE session_id = %s "
                "ORDER BY created_at DESC, id DESC LIMIT %s",
                (session_id, limit),
            ).fetchall()
        # Fetched newest-first to take the tail; flip back to chronological order.
        return [
            f"{'User' if role == 'user' else 'Assistant'}: {text}" for role, text in reversed(rows)
        ]

    def close(self) -> None:
        """Close the connection pool."""
        self._pool.close()
