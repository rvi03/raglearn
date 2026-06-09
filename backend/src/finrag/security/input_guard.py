"""Heuristic input guard: pattern-based prompt-injection / jailbreak screening.

The first, dependency-free line of defence (OWASP LLM01). It blocks the
high-confidence attack phrasings — instruction overrides, prompt exfiltration,
known jailbreak handles — while staying quiet on ordinary financial questions,
so legitimate queries are never refused. A guard *model* (Llama Guard / NeMo)
swaps in behind the same :class:`~finrag.core.interfaces.InputGuard` interface
when stronger coverage is wanted.
"""

from __future__ import annotations

import re

from finrag.core.registry import registry
from finrag.core.types import GuardVerdict

# Each entry is (category, pattern). Patterns are deliberately specific to attack
# phrasings, not topics, to keep false positives near zero on finance queries.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "prompt_injection",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b.{0,30}\b"
            r"(previous|prior|above|earlier|all)\b.{0,20}\b(instruction|prompt|rule|context|message)",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(reveal|show|print|repeat|tell me|what (is|are))\b.{0,30}"
            r"\b(your )?(system )?(prompt|instructions|guidelines|rules)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "jailbreak",
        re.compile(
            r"\b(developer mode|do anything now|\bDAN\b|jailbreak|"
            r"without any (restrictions|filters|guardrails)|"
            r"pretend you (are|have) no (rules|restrictions))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_override",
        re.compile(
            r"\byou are now\b.{0,40}\b(unrestricted|uncensored|not an? (ai|assistant))\b",
            re.IGNORECASE,
        ),
    ),
]


@registry.register("input_guard", "heuristic")
class HeuristicInputGuard:
    """An :class:`~finrag.core.interfaces.InputGuard` driven by attack patterns."""

    def inspect(self, text: str) -> GuardVerdict:
        """Block ``text`` if it matches a known injection / jailbreak pattern."""
        for category, pattern in _RULES:
            if pattern.search(text):
                return GuardVerdict(
                    allowed=False,
                    category=category,
                    reason=f"query matched a {category.replace('_', ' ')} pattern",
                )
        return GuardVerdict(allowed=True)
