"""LLM document-kind classifier — the India fallback when cover rules abstain.

When the deterministic title rules cannot name an India document's kind (a
graphic title slide, a keyword-less press-release cover), this asks the text LLM
to choose from the same canonical set. It is a fallback, not the default: it runs
only when the rules miss, and a failure (LLM down, unparseable reply, off-list
answer) returns ``None`` so the caller falls through to ``unknown`` + review
rather than crashing ingestion.
"""

from __future__ import annotations

import json
import logging
import re

from finrag.core.config import Settings
from finrag.core.errors import GenerationError
from finrag.core.interfaces.generation import LLMBackend
from finrag.core.wiring import resolve_adapter
from finrag.ingestion.identity import rules

logger = logging.getLogger(__name__)

# The closed set the LLM must choose from — the same canonical kinds the rules
# normalize to, so rule and LLM results are interchangeable downstream.
_CANONICAL = (
    rules.ANNUAL_REPORT,
    rules.FINANCIAL_RESULTS,
    rules.INVESTOR_PRESENTATION,
    rules.PRESS_RELEASE,
    rules.EARNINGS_CALL_TRANSCRIPT,
)

_PROMPT = (
    "You classify an Indian company filing from its cover-page text.\n"
    'Reply with ONLY a JSON object: {{"doc_type": <one of [{labels}] or "unknown">, '
    '"confidence": <number 0.0-1.0>}}.\n\n'
    "Cover text:\n{text}\n"
)

# The model may wrap the JSON in prose; grab the first object.
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _parse(raw: str) -> tuple[str, float] | None:
    """Parse the model's reply into (canonical doc_type, confidence), or ``None``."""
    match = _JSON_OBJECT.search(raw)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    doc_type = data.get("doc_type")
    if doc_type not in _CANONICAL:  # off-list (incl. "unknown") -> abstain
        return None
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    return doc_type, confidence


class LlmDocTypeClassifier:
    """Classifies a document's kind via the text LLM (India rules fallback)."""

    def __init__(self, backend: LLMBackend) -> None:
        """Bind the classifier to an LLM backend."""
        self._backend = backend

    def __call__(self, title_text: str) -> tuple[str, float] | None:
        """Return (doc_type, confidence) for the cover text, or ``None`` to abstain."""
        prompt = _PROMPT.format(labels=", ".join(_CANONICAL), text=title_text[:2000])
        try:
            raw = self._backend.generate(prompt).text
        except GenerationError:
            logger.warning("llm doc-type classify failed; abstaining", exc_info=True)
            return None
        return _parse(raw)


def build_doctype_classifier(settings: Settings) -> LlmDocTypeClassifier:
    """Build the LLM classifier from the config-active ``llm_backend`` (Ollama)."""
    backend = resolve_adapter(settings, "llm_backend", url=settings.services.ollama_url)
    return LlmDocTypeClassifier(backend)
