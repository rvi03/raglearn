"""Pipeline stage interfaces.

Every pluggable stage is a ``Protocol`` defined here. Adapters are
structurally typed against these - they implement the methods without inheriting
- which keeps adapters decoupled from this package.
"""

from finrag.core.interfaces.crosscutting import (
    CostModel,
    InputGuard,
    MonitorEmitter,
    OutputGuard,
    PiiRedactor,
    SpanHandle,
    Tracer,
)
from finrag.core.interfaces.generation import LLMBackend
from finrag.core.interfaces.harness import HarnessStep
from finrag.core.interfaces.ingestion import (
    Chunker,
    Embedder,
    Intake,
    MetadataExtractor,
    SourceConnector,
)
from finrag.core.interfaces.retrieval import (
    Fusion,
    QueryTransform,
    Reranker,
    Retriever,
    Router,
    StructuredQA,
    VisualRetriever,
)
from finrag.core.interfaces.storage import GraphIndex, StructuredStore, VectorStore

__all__ = [
    "Chunker",
    "CostModel",
    "Embedder",
    "Fusion",
    "GraphIndex",
    "HarnessStep",
    "InputGuard",
    "Intake",
    "LLMBackend",
    "MetadataExtractor",
    "MonitorEmitter",
    "OutputGuard",
    "PiiRedactor",
    "QueryTransform",
    "Reranker",
    "Retriever",
    "Router",
    "SourceConnector",
    "SpanHandle",
    "StructuredQA",
    "StructuredStore",
    "Tracer",
    "VectorStore",
    "VisualRetriever",
]
