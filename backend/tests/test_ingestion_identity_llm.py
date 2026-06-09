"""Tests for the India LLM doc-type classifier and its injection into dispatch."""

from __future__ import annotations

from finrag.core.errors import GenerationError
from finrag.core.types import (
    BlockKind,
    LLMResponse,
    ParsedStructure,
    RawDocument,
    StructureBlock,
    StructureSection,
)
from finrag.ingestion.identity import resolve_identity_extractor, rules
from finrag.ingestion.identity.india import IndiaIdentityExtractor
from finrag.ingestion.identity.llm_classifier import LlmDocTypeClassifier
from finrag.ingestion.identity.us import UsIdentityExtractor


class _FakeBackend:
    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(text=self._text)


class _BoomBackend:
    def generate(self, prompt: str) -> LLMResponse:
        raise GenerationError("ollama down")


def test_parses_valid_json() -> None:
    classifier = LlmDocTypeClassifier(
        _FakeBackend('{"doc_type": "financial_results", "confidence": 0.82}')
    )
    assert classifier("cover text") == (rules.FINANCIAL_RESULTS, 0.82)


def test_extracts_json_embedded_in_prose() -> None:
    classifier = LlmDocTypeClassifier(
        _FakeBackend('Sure: {"doc_type":"press_release","confidence":0.7} ok')
    )
    assert classifier("x") == (rules.PRESS_RELEASE, 0.7)


def test_rejects_offlist_label() -> None:
    classifier = LlmDocTypeClassifier(_FakeBackend('{"doc_type": "memo", "confidence": 0.9}'))
    assert classifier("x") is None


def test_rejects_unparseable_reply() -> None:
    assert LlmDocTypeClassifier(_FakeBackend("no json here"))("x") is None


def test_abstains_on_backend_error() -> None:
    assert LlmDocTypeClassifier(_BoomBackend())("x") is None


def test_dispatcher_injects_classifier_into_india() -> None:
    calls: dict[str, bool] = {}

    def classifier(_text: str) -> tuple[str, float]:
        calls["used"] = True
        return rules.FINANCIAL_RESULTS, 0.8

    extractor = resolve_identity_extractor("india/x/y.pdf", classifier=classifier)
    assert isinstance(extractor, IndiaIdentityExtractor)

    doc = RawDocument(
        doc_id="india/x/y.pdf", filename="y.pdf", content_type="application/pdf", data=b"d"
    )
    block = StructureBlock(kind=BlockKind.TEXT, text="body")
    struct = ParsedStructure(
        source_doc_id="india/x/y.pdf",
        sections=[StructureSection(title="Untemplated Heading", level=0, blocks=[block])],
    )
    ident = extractor.identify(doc, structure=struct)

    assert calls.get("used") is True  # rules missed -> LLM fallback ran
    assert ident.doc_type == rules.FINANCIAL_RESULTS


def test_dispatcher_ignores_classifier_for_us() -> None:
    extractor = resolve_identity_extractor("us/x/y.htm", classifier=lambda _t: None)
    assert isinstance(extractor, UsIdentityExtractor)
