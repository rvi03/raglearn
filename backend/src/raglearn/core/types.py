"""Domain types shared across the pipeline.

These are the data structures that flow between stages: documents in, chunks and
facts through the indexes, queries and evidence through retrieval, and an answer
with citations out. Interfaces (``core.interfaces``) are typed entirely in terms
of the models defined here.

The models are intentionally minimal; fields are added as the pipeline needs
them.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Market(StrEnum):
    """The market a filing belongs to."""

    US = "US"
    IN = "IN"


class ChunkType(StrEnum):
    """The kind of content a chunk holds, which affects how it is retrieved."""

    TEXT = "text"
    TABLE = "table"
    KPI = "kpi"


class DetectedFormat(StrEnum):
    """The coarse format a detector reports, used to route a document to a parser.

    This is the format family, not the precise media type: XBRL and any other
    XML share ``XML`` here, and telling them apart (root element, inline ``ix:``
    tags) is a later content step, not the detector's job. ``IMAGE`` covers
    raster files (PNG/JPEG/TIFF/...) whose content is only recoverable by OCR;
    it routes to the same vision parser as ``PDF``. ``UNKNOWN`` covers anything
    unrecognized and routes to quarantine.
    """

    PDF = "pdf"
    IMAGE = "image"
    HTML = "html"
    XML = "xml"
    TEXT = "text"
    UNKNOWN = "unknown"


class FactOrigin(StrEnum):
    """Where a structured financial figure came from.

    Clean XBRL is held to a higher accuracy bar than figures extracted from PDF
    tables.
    """

    XBRL = "xbrl"
    EXTRACTED = "extracted"


class DocumentMetadata(BaseModel):
    """Provenance and filter keys attached to every chunk and fact."""

    collection_id: str
    company_name: str
    ticker: str | None = None
    cik: str | None = None
    market: Market
    filing_type: str
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    section: str | None = None
    source_doc_id: str
    page: int | None = None
    char_span: tuple[int, int] | None = None
    source_revision: int = 1
    access_tags: list[str] = Field(default_factory=list)


class RawDocument(BaseModel):
    """A document as received from a connector, before parsing.

    Either ``path`` (local file) or ``data`` (in-memory bytes) is set.
    ``source_bucket`` records the object-store bucket the document came from,
    so the XBRL lane can fetch its sibling bundle files (schema, linkbases).
    """

    doc_id: str
    filename: str
    content_type: str
    path: str | None = None
    data: bytes | None = None
    source_bucket: str | None = None


class ConnectorRequest(BaseModel):
    """An ingestion request handed to a :class:`~raglearn.core.interfaces.SourceConnector`."""

    collection_id: str
    tickers: list[str] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)


class ParsedPage(BaseModel):
    """One page of a document after parsing, with merged extractor output."""

    page_no: int
    text: str = ""
    tables_markdown: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """A unit of indexed content with its provenance."""

    chunk_id: str
    text: str
    chunk_type: ChunkType
    metadata: DocumentMetadata


class FinancialFact(BaseModel):
    """A single structured figure from XBRL or an extracted table."""

    fact_id: str
    filing_id: str
    concept: str
    value: float
    unit: str
    period: str
    dimension: str | None = None
    origin: FactOrigin


class EmbeddingVector(BaseModel):
    """A hybrid embedding: dense always, sparse when the embedder produces it."""

    dense: list[float]
    sparse: dict[int, float] | None = None


class Query(BaseModel):
    """A user question plus the context needed to retrieve against it."""

    text: str
    history: list[str] = Field(default_factory=list)
    filters: dict[str, str] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """A chunk paired with a relevance score from retrieval or reranking."""

    chunk: Chunk
    score: float


class Evidence(BaseModel):
    """The assembled support for an answer: narrative chunks and structured facts."""

    chunks: list[ScoredChunk] = Field(default_factory=list)
    facts: list[FinancialFact] = Field(default_factory=list)


class Citation(BaseModel):
    """A pointer from an answer claim back to its source."""

    id: int
    source_doc_id: str
    page: int | None = None
    section: str | None = None
    period: str | None = None
    span: tuple[int, int] | None = None


class Usage(BaseModel):
    """Token counts for one LLM call, used for cost accounting."""

    tokens_in: int = 0
    tokens_out: int = 0


class CostBreakdown(BaseModel):
    """The priced result of a :class:`~raglearn.core.interfaces.CostModel` call."""

    model: str
    tokens_in: int
    tokens_out: int
    usd: float


class LLMResponse(BaseModel):
    """The output of an :class:`~raglearn.core.interfaces.LLMBackend` call."""

    text: str
    usage: Usage = Field(default_factory=Usage)


class GenerationResult(BaseModel):
    """A drafted or finalized answer with its citations and usage."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
