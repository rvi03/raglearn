"""Observability seam.

Each stage records ``{tokens, usd, latency}`` through this seam. Records go to
logs by default; additional exporters attach here without changing call sites.
"""

from finrag.observability.langfuse_exporter import (
    LangfuseSpanExporter,
    langfuse_headers,
    langfuse_otlp_endpoint,
)
from finrag.observability.monitor_emitter import (
    INGESTION_CHANNEL,
    CompositeMonitorEmitter,
    NullMonitorEmitter,
    RedisMonitorEmitter,
)
from finrag.observability.redis_exporter import RedisSpanExporter
from finrag.observability.tracer import InProcessTracer, NullTracer

__all__ = [
    "INGESTION_CHANNEL",
    "CompositeMonitorEmitter",
    "InProcessTracer",
    "LangfuseSpanExporter",
    "NullMonitorEmitter",
    "NullTracer",
    "RedisMonitorEmitter",
    "RedisSpanExporter",
    "langfuse_headers",
    "langfuse_otlp_endpoint",
]
