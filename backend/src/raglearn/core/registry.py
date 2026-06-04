"""Adapter registry.

Every pipeline stage is an interface with one or more adapters, chosen by config
. Adapters register themselves with a decorator that lives next to the
class; a stage subpackage's ``__init__`` imports its adapter modules so those
decorators run. This keeps registration local and greppable while staying
deterministic - there is no directory scanning, so an adapter never registers by
accident or fails to register silently.

  @registry.register("embedder", "bge_m3")
  class BgeM3Embedder:...

  instance = registry.create("embedder", "bge_m3")

Resolving the *active* adapter for a stage (reading config) and building the
capability matrix live in ``core.wiring`` to keep this module free of config
dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from raglearn.core.errors import AdapterNotImplementedError, RegistryError

Factory = Callable[..., Any]


@dataclass(frozen=True)
class AdapterSpec:
    """A registered adapter: its stage, name, and the callable that builds it."""

    stage: str
    name: str
    factory: Factory


class Registry:
    """Maps ``(stage, name)`` to adapter factories.

    A factory is anything callable that returns an adapter instance - typically
    the adapter class itself.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._adapters: dict[tuple[str, str], AdapterSpec] = {}

    def register(self, stage: str, name: str) -> Callable[[Factory], Factory]:
        """Return a decorator that registers an adapter under ``(stage, name)``.

        Args:
          stage: The pipeline stage, e.g. ``"embedder"``.
          name: The adapter name, e.g. ``"bge_m3"``.

        Returns:
          A decorator that records the decorated callable and returns it
          unchanged.

        Raises:
          RegistryError: If ``(stage, name)`` is already registered.
        """

        def decorator(factory: Factory) -> Factory:
            key = (stage, name)
            if key in self._adapters:
                raise RegistryError(f"adapter already registered: stage={stage!r} name={name!r}")
            self._adapters[key] = AdapterSpec(stage, name, factory)
            return factory

        return decorator

    def has(self, stage: str, name: str) -> bool:
        """Return whether an adapter is registered for ``(stage, name)``."""
        return (stage, name) in self._adapters

    def create(self, stage: str, name: str, /, **kwargs: Any) -> Any:
        """Build and return the adapter registered for ``(stage, name)``.

        Args:
          stage: The pipeline stage.
          name: The adapter name.
          **kwargs: Passed through to the adapter factory.

        Returns:
          A new adapter instance.

        Raises:
          AdapterNotImplementedError: If nothing is registered for the pair.
        """
        spec = self._adapters.get((stage, name))
        if spec is None:
            raise AdapterNotImplementedError(stage, name)
        return spec.factory(**kwargs)

    def names(self, stage: str) -> list[str]:
        """Return the sorted names of all adapters registered for a stage."""
        return sorted(name for (s, name) in self._adapters if s == stage)


# Process-wide registry. Adapters register onto this singleton at import time.
registry = Registry()
