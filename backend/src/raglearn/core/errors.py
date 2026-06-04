"""Exception hierarchy.

Every error raised by raglearn derives from :class:`RaglearnError`, so callers
can catch the whole family with one type. Domain-specific subclasses let callers
narrow when they need to.
"""

from __future__ import annotations


class RaglearnError(Exception):
    """Base class for all raglearn errors."""


class ConfigError(RaglearnError):
    """Raised when configuration is missing, malformed, or fails validation."""


class RegistryError(RaglearnError):
    """Raised on adapter registry misuse (e.g. duplicate registration)."""


class AdapterNotImplementedError(RaglearnError):
    """Raised when an adapter is selected in config but no implementation exists.

    This is the expected state for stages that are declared in the capability
    matrix but not yet built. It carries the stage and adapter name so the API
    can report exactly what is missing.
    """

    def __init__(self, stage: str, name: str) -> None:
        """Initialize with the stage and adapter name that could not be resolved.

        Args:
          stage: The pipeline stage, e.g. ``"embedder"``.
          name: The configured adapter name, e.g. ``"bge_m3"``.
        """
        self.stage = stage
        self.name = name
        super().__init__(
            f"no implementation registered for stage={stage!r} adapter={name!r} "
            f"(declared in config but not yet built)"
        )


class IngestionError(RaglearnError):
    """Raised when a document fails to ingest."""


class RetrievalError(RaglearnError):
    """Raised when retrieval fails."""


class GenerationError(RaglearnError):
    """Raised when answer generation fails."""
