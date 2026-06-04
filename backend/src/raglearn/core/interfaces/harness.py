"""Answer-quality harness interfaces.

The harness is a chain of verification and scaffolding steps that wraps
generation: each step inspects a draft answer against the retrieved evidence and
returns a possibly revised result, raising a local model to reliable, grounded
answers. Steps are composed and toggled by config.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from raglearn.core.types import Evidence, GenerationResult


@runtime_checkable
class HarnessStep(Protocol):
    """One verification/scaffolding step in the answer-quality loop.

    A step inspects a draft answer against the evidence and returns a possibly
    revised result (e.g. citation verification, self-consistency, reflection).
    """

    def apply(self, draft: GenerationResult, evidence: Evidence) -> GenerationResult:
        """Return the draft after applying this harness step."""
        ...
