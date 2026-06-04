"""Health endpoint.

Liveness for orchestration and the Compose healthcheck. Readiness probes of the
backing services are added as those clients land.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from raglearn import __version__
from raglearn.api.deps import get_settings
from raglearn.core.config import Settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """The payload returned by ``/health``."""

    status: str
    env: str
    version: str


@router.get("/health")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report liveness, the active environment, and the running version."""
    return HealthResponse(status="ok", env=settings.env.value, version=__version__)
