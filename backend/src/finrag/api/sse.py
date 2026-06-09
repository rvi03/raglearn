"""Server-Sent Events helpers.

The frontend streams pipeline progress and chat output over SSE. This
module owns the wire format so producers never hand-assemble event frames.
"""

from __future__ import annotations

import json
from typing import Any


def format_event(event: str, data: dict[str, Any]) -> str:
    """Serialize one SSE frame with the type in the ``event:`` line.

    Args:
      event: The event name, e.g. ``"token"`` or ``"pipeline"``.
      data: The JSON-serializable payload.

    Returns:
      A string ending in a blank line, ready to write to the stream.
    """
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def format_data_event(event_type: str, data: dict[str, Any]) -> str:
    """Serialize one SSE frame with the type embedded in the ``data:`` payload.

    This is the chat/`§9.4` wire format the frontend reads: its SSE reader parses
    only ``data:`` lines and discriminates on a ``type`` field in the JSON, so the
    type rides inside the payload rather than in an ``event:`` line.

    Args:
      event_type: The event type, e.g. ``"agent_step"`` or ``"token"``.
      data: The JSON-serializable payload (must not already contain ``type``).

    Returns:
      A string ending in a blank line, ready to write to the stream.
    """
    frame = {"type": event_type, **data}
    return f"data: {json.dumps(frame, separators=(',', ':'))}\n\n"
