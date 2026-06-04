"""Wiring: turn config + registry into live adapters and a capability matrix.

This is the seam between *what is configured* (``core.config``) and *what is
built* (``core.registry``). It is kept separate from both so the registry stays
config-agnostic and the config stays behaviour-agnostic.
"""

from __future__ import annotations

from typing import Any

from raglearn.core.config import Settings
from raglearn.core.errors import ConfigError
from raglearn.core.registry import Registry
from raglearn.core.registry import registry as default_registry


def resolve_adapter(
    settings: Settings,
    stage: str,
    /,
    *,
    registry: Registry = default_registry,
    **kwargs: Any,
) -> Any:
    """Build the adapter that config marks active for ``stage``.

    Args:
      settings: Loaded application settings.
      stage: The pipeline stage to resolve.
      registry: Registry to resolve against; defaults to the process registry.
      **kwargs: Passed through to the adapter factory.

    Returns:
      A new adapter instance.

    Raises:
      ConfigError: If the stage is not present in config.
      AdapterNotImplementedError: If the active adapter has no implementation.
    """
    stage_cfg = settings.adapters.get(stage)
    if stage_cfg is None:
        raise ConfigError(f"unknown stage in config: {stage!r}")
    return registry.create(stage, stage_cfg.active, **kwargs)


def capability_matrix(
    settings: Settings, *, registry: Registry = default_registry
) -> dict[str, Any]:
    """Build the capability matrix served at ``/config``.

    For each configured stage, reports the active adapter and, for every declared
    alternative, whether it is currently implemented (i.e. registered) and which
    one is active.

    Args:
      settings: Loaded application settings.
      registry: Registry to check implementations against.

    Returns:
      A JSON-serializable mapping of stage -> matrix entry.
    """
    matrix: dict[str, Any] = {}
    for stage, cfg in settings.adapters.items():
        matrix[stage] = {
            "active": cfg.active,
            "adapters": [
                {
                    "name": name,
                    "implemented": registry.has(stage, name),
                    "active": name == cfg.active,
                }
                for name in cfg.available
            ],
        }
    return matrix
