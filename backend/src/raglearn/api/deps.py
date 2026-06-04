"""Request dependencies.

Shared FastAPI dependencies. Settings are loaded once at startup and stashed on
``app.state``; routes read them through :func:`get_settings`.
"""

from __future__ import annotations

from fastapi import Request

from raglearn.core.config import Settings


def get_settings(request: Request) -> Settings:
    """Return the application settings attached to the running app.

    Args:
      request: The incoming request, used to reach ``app.state``.

    Returns:
      The :class:`Settings` loaded at startup.
    """
    settings: Settings = request.app.state.settings
    return settings
