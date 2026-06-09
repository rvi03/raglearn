"""Postgres monitor store: durable ingestion status for the corpus + monitor views.

The Redis monitor emitter is *live-only* (pub/sub, no history) — open the monitor
after a run and it is empty. This store is the durable twin: it records each
uploaded document and its pipeline progress in Postgres, so the UI can list the
corpus and show per-document status at any time (not just while ingesting).

One class plays both roles:
- as a :class:`~finrag.core.interfaces.MonitorEmitter` (``upload``/``node``/
  ``doc_done``), written by the upload endpoint and the consumer;
- as a reader (:meth:`list_uploads`) for the ``GET /ingestion/uploads`` endpoint.

Postgres (concurrent reader+writer via MVCC) lets the API read while the consumer
writes. Access goes through a small connection pool (FastAPI's sync-dependency
threadpool reads from many threads).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from psycopg_pool import ConnectionPool

_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS ingestion_documents (
        doc_id       TEXT PRIMARY KEY,
        upload_id    TEXT NOT NULL,
        country      TEXT NOT NULL,
        filename     TEXT NOT NULL,
        created      TEXT NOT NULL,
        stage        TEXT,
        stage_status TEXT,
        outcome      TEXT,
        detail       TEXT,
        updated_at   TIMESTAMP DEFAULT current_timestamp
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ingdocs_upload ON ingestion_documents (upload_id)",
    # Per-stage execution trace, so the monitor DAG (MiniDag + DocTrace) renders
    # durably after a run — not just live. One row per (document, stage); the
    # latest status/detail for that stage wins. ``seq`` preserves arrival order so
    # the trace renders in the order stages actually ran.
    """
    CREATE TABLE IF NOT EXISTS ingestion_trace (
        doc_id  TEXT NOT NULL,
        stage   TEXT NOT NULL,
        label   TEXT NOT NULL,
        status  TEXT NOT NULL,
        detail  TEXT,
        seq     BIGSERIAL,
        PRIMARY KEY (doc_id, stage)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ingtrace_doc ON ingestion_trace (doc_id, seq)",
)


class PostgresMonitorStore:
    """Durable monitor emitter + reader backed by Postgres."""

    def __init__(self, dsn: str) -> None:
        """Open a connection pool and ensure the schema exists.

        Args:
          dsn: Postgres connection string.
        """
        self._pool = ConnectionPool(dsn, min_size=1, open=True)
        with self._pool.connection() as conn:
            for statement in _SCHEMA:
                conn.execute(statement)

    # --- MonitorEmitter side (writes) -----------------------------------------

    def upload(
        self, *, upload_id: str, country: str, created: str, docs: Sequence[tuple[str, str]]
    ) -> None:
        """Record an accepted upload and its documents (status ``pending``)."""
        rows = [(doc_id, upload_id, country, filename, created) for doc_id, filename in docs]
        if not rows:
            return
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO ingestion_documents (doc_id, upload_id, country, filename, created) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (doc_id) DO UPDATE SET "
                "  upload_id = EXCLUDED.upload_id, country = EXCLUDED.country, "
                "  filename = EXCLUDED.filename, created = EXCLUDED.created, "
                "  stage = NULL, stage_status = NULL, outcome = NULL, detail = NULL, "
                "  updated_at = current_timestamp",
                rows,
            )

    def node(
        self,
        *,
        upload_id: str,
        doc_id: str,
        stage: str,
        label: str,
        status: str,
        detail: str | None = None,
    ) -> None:
        """Record a stage transition for one document: latest stage + trace node."""
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE ingestion_documents "
                "SET stage = %s, stage_status = %s, detail = %s, updated_at = current_timestamp "
                "WHERE doc_id = %s",
                (stage, status, detail, doc_id),
            )
            # Append/refresh this stage in the durable trace (keeps arrival order
            # via the seq default, which a conflict update leaves untouched).
            conn.execute(
                "INSERT INTO ingestion_trace (doc_id, stage, label, status, detail) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (doc_id, stage) DO UPDATE SET "
                "  label = EXCLUDED.label, status = EXCLUDED.status, detail = EXCLUDED.detail",
                (doc_id, stage, label, status, detail),
            )

    def doc_done(self, *, upload_id: str, doc_id: str, outcome: str) -> None:
        """Record a document's terminal outcome."""
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE ingestion_documents "
                "SET outcome = %s, updated_at = current_timestamp WHERE doc_id = %s",
                (outcome, doc_id),
            )

    # --- reader side ----------------------------------------------------------

    def list_uploads(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent uploads (newest first), each with its documents + status.

        Per-document ``status`` is the terminal ``outcome`` when known, else
        ``processing`` once a stage has started, else ``pending``.
        """
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT upload_id, country, created, doc_id, filename, stage, stage_status, "
                "outcome, detail FROM ingestion_documents ORDER BY created DESC, doc_id"
            ).fetchall()
            trace_rows = conn.execute(
                "SELECT doc_id, stage, label, status, detail "
                "FROM ingestion_trace ORDER BY doc_id, seq"
            ).fetchall()

        # Per-document execution trace (ordered) for the monitor DAG.
        traces: dict[str, list[dict[str, Any]]] = {}
        for t_doc_id, t_stage, t_label, t_status, t_detail in trace_rows:
            traces.setdefault(t_doc_id, []).append(
                {"id": t_stage, "label": t_label, "status": t_status, "detail": t_detail}
            )

        uploads: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for (
            upload_id,
            country,
            created,
            doc_id,
            filename,
            stage,
            stage_status,
            outcome,
            detail,
        ) in rows:
            up = uploads.get(upload_id)
            if up is None:
                up = {"upload_id": upload_id, "country": country, "created": created, "docs": []}
                uploads[upload_id] = up
                order.append(upload_id)
            status = outcome or ("processing" if stage else "pending")
            up["docs"].append(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "stage": stage,
                    "stage_status": stage_status,
                    "outcome": outcome,
                    "trace": traces.get(doc_id, []),
                    "status": status,
                    "detail": detail,
                }
            )
        return [uploads[uid] for uid in order[:limit]]

    def close(self) -> None:
        """Close the connection pool."""
        self._pool.close()
