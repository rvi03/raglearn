"""Document upload endpoint.

Accepts a user upload — a single file or a whole filing folder — validates it for
the chosen country, and writes each file *verbatim* into the ``filings`` bucket
under ``<country>/<upload_id>/<relative-path>``. Names and folder structure are
preserved unchanged: the XBRL bundle assembler correlates a filing's files by
their shared directory prefix and filename stem, so renaming or flattening would
break fact extraction.

Writing the objects is the whole job — each PUT raises a bucket notification that
drives the existing ingestion pipeline. Identity, dedup, and versioning are all
content-derived downstream, so this endpoint makes none of those decisions: a
re-upload simply gets a fresh ``upload_id`` and the pipeline dedups by content.
"""

from __future__ import annotations

import asyncio
import posixpath
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from finrag.api.deps import get_monitor_emitter, get_object_store
from finrag.core.interfaces.crosscutting import MonitorEmitter
from finrag.ingestion.object_store import ObjectStore

router = APIRouter(prefix="/ingest", tags=["ingest"])

# The single bucket every uploaded filing lands in; its put-notification drives
# ingestion (created with that notification by infra, not by this endpoint).
_FILINGS_BUCKET = "filings"
# Extensions an inline-XBRL instance can take (a linkbase/schema is not one).
_INSTANCE_EXTENSIONS = (".htm", ".html", ".xml")


class Country(StrEnum):
    """The market an upload is filed under; routes downstream handling."""

    US = "us"
    INDIA = "india"


class UploadedDoc(BaseModel):
    """One stored object resulting from an upload."""

    doc_id: str  # object key = <country>/<upload_id>/<relative path>
    filename: str  # relative path as uploaded
    size: int
    content_type: str


class UploadResponse(BaseModel):
    """The result of an upload: its id and the objects written."""

    upload_id: str
    country: Country
    docs: list[UploadedDoc]


def _has_complete_xbrl_bundle(relative_paths: list[str]) -> bool:
    """Return whether the upload contains an inline-XBRL instance with its schema.

    A US filing's facts need the instance (``{stem}.htm``) plus its schema
    (``{stem}.xsd``) in the same folder; this mirrors the bundle assembler's
    correlation — within any directory, is there a ``.xsd`` whose stem also has an
    instance file? Linkbases (``{stem}_lab.xml``) carry a different stem and so do
    not count as the instance.

    Args:
      relative_paths: The uploaded files' relative paths.

    Returns:
      True if at least one complete instance+schema pair is present.
    """
    by_dir: dict[str, set[str]] = {}
    for path in relative_paths:
        by_dir.setdefault(posixpath.dirname(path), set()).add(posixpath.basename(path))
    for names in by_dir.values():
        schema_stems = {n[:-4] for n in names if n.lower().endswith(".xsd")}
        for name in names:
            stem = name[: name.rfind(".")]
            if name.lower().endswith(_INSTANCE_EXTENSIONS) and stem in schema_stems:
                return True
    return False


@router.post("/upload", response_model=UploadResponse)
async def upload(
    store: Annotated[ObjectStore, Depends(get_object_store)],
    emitter: Annotated[MonitorEmitter, Depends(get_monitor_emitter)],
    country: Annotated[Country, Form()],
    files: Annotated[list[UploadFile], File()],
) -> UploadResponse:
    """Store an uploaded filing's files and return the objects written.

    Args:
      store: The object store to write into.
      emitter: Monitor emitter; announces the batch to the live DAG once stored.
      country: The market the upload is filed under (routes + validation).
      files: The uploaded files; each ``filename`` carries its relative path so
        folder uploads preserve structure.

    Returns:
      The upload's id, country, and the per-file objects written.

    Raises:
      HTTPException: The upload is empty, a file lacks a path, or a US upload is
        missing its XBRL instance/schema.
    """
    relative_paths = [file.filename or "" for file in files]
    if any(not path for path in relative_paths):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "every uploaded file must have a name")

    if country is Country.US and not _has_complete_xbrl_bundle(relative_paths):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "a US filing must include the inline-XBRL instance and its matching .xsd",
        )

    upload_id = uuid.uuid4().hex
    docs: list[UploadedDoc] = []
    for file in files:
        relative_path = file.filename or ""
        data = await file.read()
        content_type = file.content_type or "application/octet-stream"
        key = f"{country.value}/{upload_id}/{relative_path}"
        await store.put(_FILINGS_BUCKET, key, data, content_type)
        docs.append(
            UploadedDoc(
                doc_id=key, filename=relative_path, size=len(data), content_type=content_type
            )
        )

    # Announce the batch to the live monitor DAG; the consumer fills in the
    # per-document stage events as ingestion runs. The emitter's publish is sync,
    # so it runs off the event loop. Done off the loop so a slow/absent broker
    # never blocks the upload response.
    await asyncio.to_thread(
        emitter.upload,
        upload_id=upload_id,
        country=country.value,
        created=datetime.now(UTC).isoformat(),
        docs=[(doc.doc_id, doc.filename) for doc in docs],
    )

    return UploadResponse(upload_id=upload_id, country=country, docs=docs)
