"""Decoding of object-store bucket notifications.

When a file lands in the object store, the store publishes an S3-event-shaped
JSON notification to the ingestion topic. This module turns that raw message
into a clean list of :class:`S3ObjectEvent` values for the consumer to act on.

Two facts about the wire format drive the code here:

* Object keys arrive **URL-encoded** (a path separator is ``%2F``, a space is
  ``+``), following the S3 convention. They are decoded back to a plain key.
* A single notification carries a ``Records`` array; a configuration *test*
  event (sent when the notification target is first bound) has no records and
  decodes to an empty list rather than an error.
"""

from __future__ import annotations

import json
from urllib.parse import unquote_plus

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from finrag.core.errors import IngestionError


class S3ObjectEvent(BaseModel):
    """A single object-created notification, with the key already decoded."""

    event_name: str
    bucket: str
    key: str
    size: int
    content_type: str | None
    etag: str | None


class _Object(BaseModel):
    """The ``s3.object`` block of one notification record."""

    model_config = ConfigDict(populate_by_name=True)

    key: str
    size: int = 0
    content_type: str | None = Field(default=None, alias="contentType")
    etag: str | None = Field(default=None, alias="eTag")


class _Bucket(BaseModel):
    """The ``s3.bucket`` block of one notification record."""

    name: str


class _S3(BaseModel):
    """The ``s3`` payload pairing a bucket with the object that changed."""

    model_config = ConfigDict(populate_by_name=True)

    bucket: _Bucket
    obj: _Object = Field(alias="object")


class _Record(BaseModel):
    """One change record within a notification."""

    model_config = ConfigDict(populate_by_name=True)

    event_name: str = Field(alias="eventName")
    s3: _S3


class _Notification(BaseModel):
    """The notification envelope; ``Records`` is absent for a test event."""

    model_config = ConfigDict(populate_by_name=True)

    records: list[_Record] | None = Field(default=None, alias="Records")


def parse_object_events(message: bytes | str) -> list[S3ObjectEvent]:
    """Decode a bucket-notification message into object events.

    Args:
      message: The raw notification value from the ingestion topic, as bytes
        (UTF-8) or text.

    Returns:
      One :class:`S3ObjectEvent` per record, with keys URL-decoded. A test event
      (no records) yields an empty list.

    Raises:
      IngestionError: The message is not valid JSON, or its shape does not match
        a bucket notification.
    """
    text = message.decode("utf-8") if isinstance(message, bytes) else message
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise IngestionError(f"notification is not valid JSON: {exc}") from exc

    try:
        envelope = _Notification.model_validate(payload)
    except ValidationError as exc:
        raise IngestionError(f"notification has an unexpected shape: {exc}") from exc

    return [
        S3ObjectEvent(
            event_name=record.event_name,
            bucket=record.s3.bucket.name,
            key=unquote_plus(record.s3.obj.key),
            size=record.s3.obj.size,
            content_type=record.s3.obj.content_type,
            etag=record.s3.obj.etag,
        )
        for record in (envelope.records or [])
    ]
