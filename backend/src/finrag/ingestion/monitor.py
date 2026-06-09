"""Monitor helpers: pipeline stage labels and upload-key parsing.

Shared between the router (which emits stage transitions) and the upload endpoint
(which announces a batch). Keeps the stage vocabulary in one place so the labels
the monitor DAG renders stay consistent with the stages the pipeline runs.
"""

from __future__ import annotations

# Stage id -> human label, for the ``node`` events the router emits. The ids match
# what the monitor frontend collapses into its macro DAG (detect/parse/chunk/
# embed/index), so they must stay in sync with the frontend's stage map.
STAGE_LABELS: dict[str, str] = {
    "detect": "Detect format",
    "route": "Route",
    "parse": "Parse structure",
    "identify": "Identify",
    "chunk": "Chunk",
    "embed": "Embed",
    "index": "Index",
    "bundle": "Bundle check",
    "extract": "Extract facts",
    "write": "Write facts",
}


def split_upload_key(doc_id: str) -> tuple[str, str] | None:
    """Recover ``(country, upload_id)`` from an object key, or ``None`` if it can't.

    Upload objects are keyed ``<country>/<upload_id>/<relpath>`` by the ingest
    endpoint, so the country and upload id are the first two path segments. A key
    that does not have at least those two segments is not a monitored upload.
    """
    parts = doc_id.split("/", 2)
    if len(parts) < 3:
        return None
    return parts[0], parts[1]
