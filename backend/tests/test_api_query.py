"""Tests for the /query endpoint (hermetic: the answer service is faked)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from finrag.api.deps import get_query_service
from finrag.core.types import Citation, Query, Usage
from finrag.retrieval.answer import GroundedAnswer, SourceCard


class _FakeService:
    """Records the query it was asked and returns a canned grounded answer."""

    def __init__(self, result: GroundedAnswer) -> None:
        self.result = result
        self.query: Query | None = None

    def answer(self, query: Query) -> GroundedAnswer:
        self.query = query
        return self.result


def _result() -> GroundedAnswer:
    return GroundedAnswer(
        answer="Net sales were $123.4B [1].",
        citations=[Citation(id=1, source_doc_id="us/a.htm", page=31)],
        sources=[
            SourceCard(
                id=1,
                chunk_id="a-0",
                title="Mock Corp · 10-K",
                url="/sources/us/a.htm#p31",
                snippet="…",
            )
        ],
        usage=Usage(tokens_in=10, tokens_out=5),
    )


@pytest.fixture
def service() -> _FakeService:
    return _FakeService(_result())


@pytest.fixture
def query_client(client: TestClient, service: _FakeService) -> Iterator[TestClient]:
    client.app.dependency_overrides[get_query_service] = lambda: service
    yield client
    client.app.dependency_overrides.clear()


def test_query_returns_answer_with_citations_and_sources(
    query_client: TestClient, service: _FakeService
) -> None:
    response = query_client.post(
        "/query", json={"text": "What were net sales?", "filters": {"collection_id": "c1"}}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Net sales were $123.4B [1]."
    assert body["citations"] == [
        {
            "id": 1,
            "source_doc_id": "us/a.htm",
            "page": 31,
            "section": None,
            "period": None,
            "span": None,
        }
    ]
    assert body["sources"][0]["title"] == "Mock Corp · 10-K"
    # the request reached the service intact
    assert service.query is not None
    assert service.query.text == "What were net sales?"
    assert service.query.filters == {"collection_id": "c1"}


def test_query_defaults_filters_to_empty(query_client: TestClient, service: _FakeService) -> None:
    response = query_client.post("/query", json={"text": "hi"})

    assert response.status_code == 200
    assert service.query is not None
    assert service.query.filters == {}


def test_query_requires_text(query_client: TestClient) -> None:
    response = query_client.post("/query", json={"filters": {}})

    assert response.status_code == 422  # missing required `text`
