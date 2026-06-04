"""Config endpoint.

Serves the capability matrix: every stage, its active adapter, and which
declared adapters are actually implemented. This is how the UI documents what is
built versus comparable.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from raglearn.api.deps import get_settings
from raglearn.core.config import Settings
from raglearn.core.wiring import capability_matrix

router = APIRouter(tags=["config"])


@router.get("/config")
def get_config(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, Any]:
    """Return the capability matrix for the active configuration."""
    return capability_matrix(settings)
