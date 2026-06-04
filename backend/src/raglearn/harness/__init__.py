"""Answer-quality harness adapters.

The verification and scaffolding steps that wrap generation: citation
verification, self-consistency, reflection, judge-gate, numeric checks. This
package grows as new techniques are added; import each step module here so its
``@register`` decorator runs.
"""

__all__: list[str] = []
