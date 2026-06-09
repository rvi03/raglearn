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


class NumericAuthority(StrEnum):
    """How much a figure taken from a document can be trusted.

    Derived from the document's kind, not classified directly: audited periodic
    filings (10-K/10-Q; Indian annual reports and financial results) are
    ``AUTHORITATIVE``; issuer commentary that is selective or unaudited
    (presentations, press releases, earnings-call transcripts) is ``INDICATIVE``.
    """

    AUTHORITATIVE = "authoritative"
    INDICATIVE = "indicative"


class DocumentIdentity(BaseModel):
    """The content-derived identity of a document, independent of its filename.

    Produced by an :class:`~finrag.core.interfaces.IdentityExtractor`. The path
    contributes only ``market`` (the enforced ``filings/<country>/`` prefix);
    everything else comes from the document's content. ``logical_key`` groups all
    versions of the *same* logical document (so an amendment/revision can be
    recognized), and ``content_hash`` distinguishes one version from another —
    together they form the versioned primary key carried downstream.
    """

    market: Market
    collection_id: str
    logical_key: str
    content_hash: str
    doc_type: str
    numeric_authority: NumericAuthority
    company: str | None = None
    cik: str | None = None
    ticker: str | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    recency: str | None = None
    confidence: float = 1.0
    needs_review: bool = False


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
    # Versioned-identity fields, populated from DocumentIdentity. ``content_hash``
    # is the version key and ``logical_key`` groups versions of one logical
    # document; together they form the versioned primary key the stores carry.
    # Optional so pre-identity construction paths (and tests) still validate.
    content_hash: str | None = None
    logical_key: str | None = None
    numeric_authority: NumericAuthority | None = None
    recency: str | None = None


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
    """An ingestion request handed to a :class:`~finrag.core.interfaces.SourceConnector`."""

    collection_id: str
    tickers: list[str] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)


class BlockKind(StrEnum):
    """The kind of a parsed content block within a section."""

    TEXT = "text"
    TABLE = "table"


class StructureBlock(BaseModel):
    """One contiguous piece of content inside a section: prose or a whole table.

    ``text`` holds the prose for a ``TEXT`` block and the rendered markdown for a
    ``TABLE`` block. ``page`` is the source page when the parser provides it
    (Docling's vision pipeline does; the SEC HTML parser generally does not).
    """

    kind: BlockKind
    text: str
    page: int | None = None


class StructureSection(BaseModel):
    """A titled span of a document: a heading and the blocks beneath it.

    ``title`` is ``None`` for the preamble (content before the first heading).
    ``level`` is the heading depth: ``0`` for a top-level section (a filing's
    ``Part``/``Item``), higher for nested subsections.
    """

    title: str | None
    level: int
    blocks: list[StructureBlock] = Field(default_factory=list)


class ParsedStructure(BaseModel):
    """A document reduced to its section structure, independent of the parser.

    Both parse lanes normalize into this one shape — the SEC HTML parser
    (sec-parser) and the PDF parser (Docling vision) — so the chunker consumes a
    single representation and never depends on a parsing library's own types.
    """

    source_doc_id: str
    sections: list[StructureSection] = Field(default_factory=list)


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


class CollectionMetadata(BaseModel):
    """A company-level collection (one company = one collection namespace).

    Sourced from a filing's XBRL DEI facts. Fields are nullable because older
    filings omit some DEI (e.g. ``TradingSymbol``).
    """

    collection_id: str
    company: str | None = None
    ticker: str | None = None
    cik: str | None = None
    market: Market
    status: str = "ingested"


class FilingMetadata(BaseModel):
    """A filing-level row, the parent of its financial facts.

    ``filed_date`` (EDGAR submission header) and ``source_url`` (connector) are
    not in the XBRL instance, so they stay ``None`` until those sources land.
    ``version`` defaults to 1; supersede logic (10-K/A) is the dedup step's job.
    """

    filing_id: str
    collection_id: str
    filing_type: str | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    filed_date: str | None = None
    source_url: str | None = None
    version: int = 1
    # Groups all versions of one logical filing (company x period x form family),
    # so an amendment/restatement can be recognized. Set from DocumentIdentity.
    logical_key: str | None = None


class XbrlExtraction(BaseModel):
    """The full result of extracting one XBRL filing: its metadata and its facts.

    A single Arelle load yields all three: the collection and filing rows (from
    DEI) and the numeric facts. They are written together so a fact always has
    its parent ``filings`` row (the foreign key holds).
    """

    collection: CollectionMetadata
    filing: FilingMetadata
    facts: list[FinancialFact]


class EmbeddingVector(BaseModel):
    """A hybrid embedding: dense always, sparse when the embedder produces it."""

    dense: list[float]
    sparse: dict[int, float] | None = None


class Query(BaseModel):
    """A user question plus the context needed to retrieve against it."""

    text: str
    history: list[str] = Field(default_factory=list)
    filters: dict[str, str] = Field(default_factory=dict)
    # The caller's granted access tags; retrieval returns only public chunks
    # (untagged) plus chunks carrying one of these. From the authenticated
    # principal in production; empty means public-only.
    access_tags: list[str] = Field(default_factory=list)


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


class GuardVerdict(BaseModel):
    """The result of a security guard inspecting a query or an answer."""

    allowed: bool = True
    # What tripped the guard, e.g. ``"prompt_injection"`` or ``"jailbreak"``.
    category: str | None = None
    reason: str | None = None


class Redaction(BaseModel):
    """The result of a :class:`~finrag.core.interfaces.PiiRedactor` pass."""

    text: str  # the input with any detected PII masked
    # Entity types that were masked, e.g. ``["EMAIL", "IN_PAN"]`` (deduped, sorted).
    entities: list[str] = Field(default_factory=list)


class CostBreakdown(BaseModel):
    """The priced result of a :class:`~finrag.core.interfaces.CostModel` call."""

    model: str
    tokens_in: int
    tokens_out: int
    usd: float


class Span(BaseModel):
    """One timed unit of work in a trace tree.

    A :class:`~finrag.core.interfaces.Tracer` records a span per stage
    (retrieve, rerank, generate, harness) nested under a root span for the whole
    query. Token usage is captured on the span that made the LLM call and priced
    into ``cost`` when the trace closes; the roll-up properties aggregate the
    subtree so the query total is one read off the root.
    """

    name: str
    status: str = "ok"  # "ok" | "error"
    latency_ms: float = 0.0
    usage: Usage = Field(default_factory=Usage)
    # The model whose call this span timed, if any; set when usage is recorded so
    # the tracer can price it. ``None`` for non-LLM spans (e.g. a vector search).
    model: str | None = None
    # Priced at trace close from ``usage`` and ``model`` via the CostModel.
    cost: CostBreakdown | None = None
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)
    children: list[Span] = Field(default_factory=list)

    @property
    def total_usd(self) -> float:
        """Sum of this span's cost and every descendant's."""
        own = self.cost.usd if self.cost is not None else 0.0
        return own + sum(child.total_usd for child in self.children)

    @property
    def total_usage(self) -> Usage:
        """Summed token usage across this span and every descendant."""
        tokens_in = self.usage.tokens_in
        tokens_out = self.usage.tokens_out
        for child in self.children:
            sub = child.total_usage
            tokens_in += sub.tokens_in
            tokens_out += sub.tokens_out
        return Usage(tokens_in=tokens_in, tokens_out=tokens_out)


class LLMResponse(BaseModel):
    """The output of an :class:`~finrag.core.interfaces.LLMBackend` call."""

    text: str
    usage: Usage = Field(default_factory=Usage)
    # The model that produced this response, carried alongside usage so the
    # tracer can price the call. Empty when the backend does not report it.
    model: str = ""


class GenerationResult(BaseModel):
    """A drafted or finalized answer with its citations and usage."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    # Set by the answer-quality harness: how well the answer is supported by its
    # cited evidence, in [0, 1]. None until a harness step has scored it.
    grounding_confidence: float | None = None
