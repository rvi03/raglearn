"""Adapter discovery.

Importing a stage subpackage runs the ``@register`` decorators on its adapters.
:func:`load_adapters` imports every stage subpackage so the registry is fully
populated before the app serves traffic. This is the explicit, deterministic
alternative to scanning the filesystem for adapters.
"""

from __future__ import annotations


def load_adapters() -> None:
    """Import all stage subpackages to register their adapters.

    Safe to call repeatedly: Python caches imports, so each adapter registers
    exactly once.
    """
    from raglearn import (  # noqa: F401 (imports register adapters as a side effect)
        cost,
        generation,
        harness,
        ingestion,
        retrieval,
        security,
        stores,
    )
