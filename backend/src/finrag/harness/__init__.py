"""Answer-quality harness adapters.

The verification and scaffolding steps that wrap generation: citation
verification, self-consistency, reflection, judge-gate, numeric checks. This
package grows as new techniques are added; import each step module here so its
``@register`` decorator runs.
"""

from finrag.harness import citation_verify  # imports register the adapter

__all__ = ["citation_verify"]
