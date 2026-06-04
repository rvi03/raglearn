"""Server-Sent Events helpers.

The frontend streams pipeline progress and chat output over SSE. This
module owns the wire format so producers never hand-assemble event frames.
"""

from __future__ import annotations

import json
from typing import Any


def format_event(event: str, data: dict[str, Any]) -> str:
    """Serialize one SSE frame.

    Args:
      event: The event name, e.g. ``"token"`` or ``"pipeline"``.
      data: The JSON-serializable payload.

    Returns:
      A string ending in a blank line, ready to write to the stream.
    """
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
