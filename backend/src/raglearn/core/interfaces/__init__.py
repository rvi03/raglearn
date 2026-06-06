"""Pipeline stage interfaces.

Every pluggable stage is a ``Protocol`` defined here. Adapters are
structurally typed against these - they implement the methods without inheriting
- which keeps adapters decoupled from this package.
"""

from raglearn.core.interfaces.crosscutting import CostModel
from raglearn.core.interfaces.generation import LLMBackend
from raglearn.core.interfaces.harness import HarnessStep
from raglearn.core.interfaces.ingestion import (
    Chunker,
    DocumentParser,
    Embedder,
    Intake,
    MetadataExtractor,
    SourceConnector,
)
from raglearn.core.interfaces.retrieval import (
    Fusion,
    QueryTransform,
    Reranker,
    Retriever,
    Router,
    StructuredQA,
    VisualRetriever,
)
from raglearn.core.interfaces.storage import GraphIndex, StructuredStore, VectorStore

__all__ = [
    "Chunker",
    "CostModel",
    "DocumentParser",
    "Embedder",
    "Fusion",
    "GraphIndex",
    "HarnessStep",
    "Intake",
    "LLMBackend",
    "MetadataExtractor",
    "QueryTransform",
    "Reranker",
    "Retriever",
    "Router",
    "SourceConnector",
    "StructuredQA",
    "StructuredStore",
    "VectorStore",
    "VisualRetriever",
]
