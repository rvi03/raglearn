"""Logging setup.

A single place to configure process-wide logging. The format is intentionally
plain; structured/OTel export attaches later without changing call sites
(callers only ever use :func:`get_logger`).
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger to write to stdout at the given level.

    Idempotent: replaces existing handlers so repeated calls (tests, app
    factory) do not stack duplicate output.

    Args:
      level: A standard logging level name, e.g. ``"INFO"`` or ``"DEBUG"``.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name.

    Args:
      name: Usually ``__name__`` of the calling module.

    Returns:
      A standard library logger.
    """
    return logging.getLogger(name)
