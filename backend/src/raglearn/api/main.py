"""Application factory.

Builds the FastAPI app: load settings, configure logging, register adapters,
mount routes. ``uvicorn raglearn.api.main:app`` serves the module-level ``app``;
tests call :func:`create_app` with explicit settings for isolation.
"""

from __future__ import annotations

from fastapi import FastAPI

from raglearn import __version__
from raglearn.core.bootstrap import load_adapters
from raglearn.core.config import Settings, load_settings
from raglearn.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure a FastAPI application.

    Args:
      settings: Settings to run with; loaded from config files when omitted.

    Returns:
      A configured :class:`~fastapi.FastAPI` instance.
    """
    settings = settings or load_settings()
    configure_logging(settings.log_level)
    load_adapters()

    app = FastAPI(title="raglearn", version=__version__)
    app.state.settings = settings

    # Imported here so adapter registration (above) is complete first.
    from raglearn.api.routes import config, health

    app.include_router(health.router)
    app.include_router(config.router)

    logger.info("raglearn %s started (env=%s)", __version__, settings.env.value)
    return app


app = create_app()
