"""Heuristic output guard: pattern-based screening of the generated answer.

The output-side counterpart of the input guard (OWASP LLM02/LLM06). Where the
input guard stops a malicious *question*, this stops a compromised *answer* — one
that leaks the grounding scaffold, parrots its own instructions, or carries an
unsafe instruction the model was tricked into emitting. It stays quiet on ordinary
financial prose so legitimate answers are never refused. A guard *model* swaps in
behind the same :class:`~finrag.core.interfaces.OutputGuard` interface when
stronger coverage is wanted; PII redaction is a separate, complementary stage.

The same :meth:`screen` vets either the whole answer (``/query``) or a single
streamed segment (``/chat``), so the wire path and the batch path share one rule
set and can never disagree on a verdict.
"""

from __future__ import annotations

import re

from finrag.core.registry import registry
from finrag.core.types import GuardVerdict

# Each entry is (category, pattern). Patterns target the *leakage* phrasings that
# only appear when an answer has gone wrong — the verbatim grounding-prompt
# scaffold or an echoed instruction — not finance topics, to keep false positives
# near zero on real answers.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "system_prompt_leak",
        re.compile(
            r"you are a financial-analysis assistant"
            r"|answer the question using only the numbered sources"
            r"|do not use outside knowledge"
            r"|cite every claim with its source number",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_echo",
        re.compile(
            r"\b(my|the) (system )?(prompt|instructions|guidelines) (are|is|say)\b"
            r"|here (are|is) (my|the) (system )?(prompt|instructions)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "unsafe_content",
        re.compile(
            r"\b(ignore (all|the) (previous|above) instructions"
            r"|i have been (jailbroken|hacked)"
            r"|developer mode (enabled|activated))\b",
            re.IGNORECASE,
        ),
    ),
]


@registry.register("output_guard", "heuristic")
class HeuristicOutputGuard:
    """An :class:`~finrag.core.interfaces.OutputGuard` driven by leak patterns."""

    def screen(self, text: str) -> GuardVerdict:
        """Block ``text`` if it leaks the prompt scaffold or carries unsafe content."""
        for category, pattern in _RULES:
            if pattern.search(text):
                return GuardVerdict(
                    allowed=False,
                    category=category,
                    reason=f"answer matched a {category.replace('_', ' ')} pattern",
                )
        return GuardVerdict(allowed=True)
