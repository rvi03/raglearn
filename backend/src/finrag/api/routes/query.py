"""Query endpoint: ask a question, get a grounded, cited answer.

The narrative retrieval slice's entry point. A question (plus optional metadata
filters) goes in; the answer service retrieves evidence, reranks it, and has the
LLM draft an answer with ``[n]`` citations. The response carries the answer, the
citations (each pointing back to a source document and page), and the evidence
source cards the ``[n]`` markers refer to.

Non-streaming by design for this slice — a single request, a single JSON answer.
The streaming ``/chat`` (the §9.4 SSE contract the frontend codes against) lands
with the full generation vertical.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from finrag.api.deps import get_query_service
from finrag.core.types import Query
from finrag.retrieval.answer import GroundedAnswer
from finrag.retrieval.query_service import QueryService

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    """A question to answer over the indexed corpus."""

    text: str
    filters: dict[str, str] = Field(default_factory=dict)  # metadata narrowing, e.g. collection_id
    # The caller's access tags. PLACEHOLDER: in production these are derived from
    # the authenticated principal server-side, never trusted from the client.
    access_tags: list[str] = Field(default_factory=list)


@router.post("/query", response_model=GroundedAnswer)
def query(
    request: QueryRequest,
    service: Annotated[QueryService, Depends(get_query_service)],
) -> GroundedAnswer:
    """Answer a question, routing it to the exact or narrative path.

    Args:
      request: The question and optional metadata filters.
      service: The query service (router → exact figures or narrative answer).

    Returns:
      The grounded answer, its citations, the evidence sources, and token usage.
    """
    return service.answer(
        Query(text=request.text, filters=request.filters, access_tags=request.access_tags)
    )
