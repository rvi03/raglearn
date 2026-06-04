"""Observability seam.

Each stage records ``{tokens, usd, latency}`` through this seam. Records go to
logs by default; additional exporters attach here without changing call sites.
"""

__all__: list[str] = []
