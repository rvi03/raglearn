"""Request dependencies.

Shared FastAPI dependencies. Settings are loaded once at startup and stashed on
``app.state``; routes read them through :func:`get_settings`.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Request

from finrag.core.config import Settings
from finrag.core.errors import ConfigError
from finrag.core.interfaces.crosscutting import MonitorEmitter, OutputGuard, PiiRedactor, Tracer
from finrag.core.registry import registry
from finrag.core.types import Span
from finrag.core.wiring import build_structured_store, resolve_adapter
from finrag.ingestion.object_store import ObjectStore
from finrag.observability import (
    CompositeMonitorEmitter,
    LangfuseSpanExporter,
    RedisMonitorEmitter,
    RedisSpanExporter,
    langfuse_headers,
    langfuse_otlp_endpoint,
)
from finrag.retrieval.answer import AnswerService
from finrag.retrieval.composite_qa import build_structured_qa
from finrag.retrieval.fusing_retriever import FusingRetriever
from finrag.retrieval.query_service import QueryService
from finrag.stores.postgres_chat import PostgresChatStore
from finrag.stores.postgres_monitor import PostgresMonitorStore


def get_settings(request: Request) -> Settings:
    """Return the application settings attached to the running app.

    Args:
      request: The incoming request, used to reach ``app.state``.

    Returns:
      The :class:`Settings` loaded at startup.
    """
    settings: Settings = request.app.state.settings
    return settings


def get_object_store(request: Request) -> ObjectStore:
    """Return a process-wide object store, built once from settings.

    The MinIO client is cheap to construct and connects lazily, so it is created
    on first use and cached on ``app.state``. Tests override this dependency with
    a fake to keep the upload route hermetic.

    Args:
      request: The incoming request, used to reach ``app.state``.

    Returns:
      The shared :class:`ObjectStore`.
    """
    store: ObjectStore | None = getattr(request.app.state, "object_store", None)
    if store is None:
        services = request.app.state.settings.services
        store = ObjectStore(
            services.minio_endpoint, services.minio_access_key, services.minio_secret_key
        )
        request.app.state.object_store = store
    return store


def _build_trace_sinks(settings: Settings) -> list[Callable[[Span], None]]:
    """Build the configured trace exporters from settings.

    Each name in ``observability.exporters`` is wired with its runtime resources
    (a Redis client, the Langfuse OTLP endpoint). Unknown names fail loudly so a
    typo does not silently drop traces.

    Raises:
      ConfigError: For an unknown exporter name.
    """
    sinks: list[Callable[[Span], None]] = []
    for name in settings.observability.exporters:
        if name == "redis":
            import redis

            client = redis.Redis.from_url(settings.services.redis_url)
            sinks.append(RedisSpanExporter(client))
        elif name == "langfuse":
            endpoint = langfuse_otlp_endpoint(settings.services.langfuse_url)
            sinks.append(LangfuseSpanExporter.build(endpoint=endpoint, headers=langfuse_headers()))
        else:
            raise ConfigError(f"unknown trace exporter: {name!r}")
    return sinks


def get_tracer(request: Request) -> Tracer:
    """Return a process-wide tracer, built once from settings.

    Composes the active ``cost_model`` into the ``local`` tracer so spans price
    their token usage, and fans out to the configured exporters
    (``observability.exporters``). Shared by the answer and query services so a
    query's stage spans nest under one root trace. Cached on ``app.state``.

    Args:
      request: The incoming request, used to reach ``app.state``.

    Returns:
      The shared :class:`Tracer`.
    """
    tracer: Tracer | None = getattr(request.app.state, "tracer", None)
    if tracer is None:
        settings = request.app.state.settings
        cost_model = resolve_adapter(settings, "cost_model")
        tracer = resolve_adapter(
            settings, "tracer", cost_model=cost_model, sinks=_build_trace_sinks(settings)
        )
        request.app.state.tracer = tracer
    return tracer


def get_answer_service(request: Request) -> AnswerService:
    """Return a process-wide narrative answer service, built once from settings.

    Composes the active ``retriever``/``reranker``/``llm_backend`` adapters (and
    the ``embedder``/``vector_store`` the retriever needs) into the answer path.
    The heavy models behind these load lazily on first use, so construction is
    cheap; the assembled service is cached on ``app.state`` and reused. Tests
    override this dependency with a fake to stay hermetic.

    Args:
      request: The incoming request, used to reach ``app.state``.

    Returns:
      The shared :class:`AnswerService`.
    """
    service: AnswerService | None = getattr(request.app.state, "answer_service", None)
    if service is None:
        settings = request.app.state.settings
        embedder = resolve_adapter(settings, "embedder")
        vector_store = resolve_adapter(settings, "vector_store", url=settings.services.qdrant_url)
        retriever = resolve_adapter(
            settings, "retriever", embedder=embedder, vector_store=vector_store
        )
        # When fusion is active, add a lexical (BM25) arm over the same corpus and
        # fuse the two rankings (RRF) + reconcile versions. The lexical index is
        # seeded once from the store at startup; newly-ingested chunks join it on
        # the next restart.
        if settings.adapters["fusion"].active != "none":
            lexical = registry.create("retriever", "lexical", corpus=vector_store.scroll())
            retriever = FusingRetriever(
                retrievers=[retriever, lexical],
                fusion=resolve_adapter(settings, "fusion"),
                tracer=get_tracer(request),
            )
        reranker = resolve_adapter(settings, "reranker")
        llm = resolve_adapter(settings, "llm_backend", url=settings.services.ollama_url)
        harness = resolve_adapter(settings, "harness", llm=llm)
        service = AnswerService(
            retriever=retriever,
            reranker=reranker,
            llm=llm,
            harness=harness,
            tracer=get_tracer(request),
        )
        request.app.state.answer_service = service
    return service


def get_query_service(request: Request) -> QueryService:
    """Return a process-wide query service, built once from settings.

    Composes the router, the structured (exact) QA over the DuckDB store, and the
    narrative answer service into the routed entry point used by ``/query``. The
    DuckDB store is the read side here (the consumer is the single writer).
    Cached on ``app.state``; tests override this dependency with a fake.

    Args:
      request: The incoming request, used to reach ``app.state``.

    Returns:
      The shared :class:`QueryService`.
    """
    service: QueryService | None = getattr(request.app.state, "query_service", None)
    if service is None:
        settings = request.app.state.settings
        narrative = get_answer_service(request)
        structured_store = build_structured_store(settings)
        router = resolve_adapter(settings, "router")
        llm = resolve_adapter(settings, "llm_backend", url=settings.services.ollama_url)
        structured_qa = build_structured_qa(settings, store=structured_store, llm=llm)
        query_transform = resolve_adapter(settings, "query_transform", llm=llm)
        input_guard = resolve_adapter(settings, "input_guard")
        output_guard = resolve_adapter(settings, "output_guard")
        pii_redactor = resolve_adapter(settings, "pii")
        service = QueryService(
            router=router,
            structured_qa=structured_qa,
            narrative=narrative,
            store=structured_store,
            query_transform=query_transform,
            tracer=get_tracer(request),
            input_guard=input_guard,
            output_guard=output_guard,
            pii_redactor=pii_redactor,
        )
        request.app.state.query_service = service
    return service


def get_output_guard(request: Request) -> OutputGuard:
    """Return a process-wide output guard, built once from settings.

    The ``/chat`` stream uses it to screen segments live (the same guard the
    :class:`QueryService` applies to the whole answer), so the streamed text and
    the final answer are vetted by one rule set. Stateless and cheap, it is built
    on first use and cached on ``app.state``.

    Args:
      request: The incoming request, used to reach ``app.state``.

    Returns:
      The shared :class:`OutputGuard`.
    """
    guard: OutputGuard | None = getattr(request.app.state, "output_guard", None)
    if guard is None:
        guard = resolve_adapter(request.app.state.settings, "output_guard")
        request.app.state.output_guard = guard
    return guard


def get_pii_redactor(request: Request) -> PiiRedactor:
    """Return a process-wide PII redactor, built once from settings.

    The ``/chat`` stream uses it to mask high-confidence identifiers segment by
    segment (the same redactor the :class:`QueryService` applies to the whole
    answer). Stateless and cheap, it is built on first use and cached.

    Args:
      request: The incoming request, used to reach ``app.state``.

    Returns:
      The shared :class:`PiiRedactor`.
    """
    redactor: PiiRedactor | None = getattr(request.app.state, "pii_redactor", None)
    if redactor is None:
        redactor = resolve_adapter(request.app.state.settings, "pii")
        request.app.state.pii_redactor = redactor
    return redactor


def get_monitor_store(request: Request) -> PostgresMonitorStore:
    """Return a process-wide durable monitor store, built once from settings.

    Backs the corpus/monitor views with persisted per-document ingestion status
    (Postgres). Used both as a durable emitter sink (inside
    :func:`get_monitor_emitter`) and as the reader for ``GET /ingestion/uploads``.
    Cached on ``app.state``.

    Args:
      request: The incoming request, used to reach ``app.state``.

    Returns:
      The shared :class:`PostgresMonitorStore`.
    """
    store: PostgresMonitorStore | None = getattr(request.app.state, "monitor_store", None)
    if store is None:
        store = PostgresMonitorStore(request.app.state.settings.services.postgres_dsn)
        request.app.state.monitor_store = store
    return store


def get_chat_store(request: Request) -> PostgresChatStore:
    """Return a process-wide chat store, built once from settings.

    Backs Claude-style named, persistent conversations (Postgres): the sessions
    API reads/writes it, and ``/chat`` appends turns and reads recent ones back as
    short-term memory. Cached on ``app.state``.

    Args:
      request: The incoming request, used to reach ``app.state``.

    Returns:
      The shared :class:`PostgresChatStore`.
    """
    store: PostgresChatStore | None = getattr(request.app.state, "chat_store", None)
    if store is None:
        store = PostgresChatStore(request.app.state.settings.services.postgres_dsn)
        request.app.state.chat_store = store
    return store


def get_monitor_emitter(request: Request) -> MonitorEmitter:
    """Return a process-wide monitor emitter, built once from settings.

    Fans ingestion events out to two sinks: the **durable** Postgres monitor store
    (always — this is what the corpus/monitor views read back), and the **live**
    Redis pub/sub feed when ``redis`` is among ``observability.exporters`` (the DAG
    streams from it). The ``/ingest`` endpoint uses it to announce the ``upload``
    event; the consumer wires its own for the per-stage events. Cached on
    ``app.state``; thread-safe (pooled clients) for the handler threadpool.

    Args:
      request: The incoming request, used to reach ``app.state``.

    Returns:
      The shared :class:`MonitorEmitter`.
    """
    emitter: MonitorEmitter | None = getattr(request.app.state, "monitor_emitter", None)
    if emitter is None:
        settings = request.app.state.settings
        sinks: list[MonitorEmitter] = [get_monitor_store(request)]  # durable, always on
        if "redis" in settings.observability.exporters:
            import redis

            sinks.append(RedisMonitorEmitter(redis.Redis.from_url(settings.services.redis_url)))
        emitter = CompositeMonitorEmitter(sinks)
        request.app.state.monitor_emitter = emitter
    return emitter
