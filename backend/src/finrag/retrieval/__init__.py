"""Retrieval-plane adapters.

Query transforms, router, retrievers, reranker, fusion, and structured QA live
here. Import each adapter module here so its ``@register`` decorator runs.
"""

from finrag.retrieval import (  # imports register the adapters
    fusion,
    lexical,
    query_transform,
    reranker,
    retriever,
    router,
    structured_qa,
    text_to_sql,
)

__all__ = [
    "fusion",
    "lexical",
    "query_transform",
    "reranker",
    "retriever",
    "router",
    "structured_qa",
    "text_to_sql",
]
