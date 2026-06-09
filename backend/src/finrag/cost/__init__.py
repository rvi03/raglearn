"""Cost-model adapters.

Importing the adapter modules here is what registers them - the explicit import
is the deterministic half of the registry pattern (see ``core.registry``).
"""

from finrag.cost import local  # re-exported; the import registers the adapter

__all__ = ["local"]
