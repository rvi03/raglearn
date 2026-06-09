"""Source document endpoint: stream an uploaded filing back for the viewer.

The chat citations link to ``/sources/<doc_id>`` ("Open full document"); this
serves those bytes from the object store (the ``filings`` bucket the upload
endpoint wrote to), so a cited passage can be opened in its original document.
Read-only: it fetches by the object key and streams it with a best-effort media
type. The ``#p<page>`` fragment in the link is resolved client-side by the
browser's viewer, so it never reaches here.
"""

from __future__ import annotations

import posixpath
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response

from finrag.api.deps import get_object_store
from finrag.ingestion.object_store import ObjectStore

router = APIRouter(tags=["sources"])

# The bucket the upload endpoint writes filings to (mirrors ingest.py).
_FILINGS_BUCKET = "filings"

# Best-effort media type so the browser renders inline instead of downloading.
_MEDIA: dict[str, str] = {
    ".pdf": "application/pdf",
    ".htm": "text/html",
    ".html": "text/html",
    ".xml": "application/xml",
    ".txt": "text/plain",
}


@router.get("/sources/{doc_id:path}")
async def get_source(
    doc_id: str, store: Annotated[ObjectStore, Depends(get_object_store)]
) -> Response:
    """Stream a stored filing by its object key (``<country>/<upload_id>/<relpath>``).

    Args:
      doc_id: The object key to fetch (the citation's ``source_doc_id``).
      store: The object store to read from.

    Returns:
      The document bytes with an inline media type.

    Raises:
      HTTPException: 404 if the object can't be fetched.
    """
    try:
        data = await store.fetch(_FILINGS_BUCKET, doc_id)
    except Exception as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"source not found: {doc_id}") from exc
    media = _MEDIA.get(posixpath.splitext(doc_id)[1].lower(), "application/octet-stream")
    filename = posixpath.basename(doc_id)
    return Response(
        content=data,
        media_type=media,
        headers={"Content-Disposition": f"inline; filename={filename}"},
    )
