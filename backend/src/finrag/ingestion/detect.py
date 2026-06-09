"""Content-based format detection backed by an Apache Tika server.

The format we route on comes from the bytes, never the filename or the
uploader's declared content type - both are untrusted. Detection only needs the
file head (magic numbers, document structure), so we send a leading prefix
rather than the whole object, which keeps large filings off the wire just to
learn their type.

Caveat: ZIP-container formats (``.xlsx``, ``.docx``) record their type in a
trailer at the end of the file, so a head prefix cannot identify them. The
corpus here is PDF/HTML/XBRL/text, all head-detectable; supporting Office
formats later means sending the full bytes, not a prefix.
"""

from __future__ import annotations

import httpx

from finrag.core.errors import IngestionError
from finrag.core.registry import registry
from finrag.core.types import DetectedFormat

# Leading bytes sent to Tika. Enough to carry the magic number and opening
# structure of every format we route on, while keeping large objects off the
# wire (see the module caveat on trailer-typed formats).
_DETECT_PREFIX_BYTES = 8192

# Tika media types mapped to the coarse format the router switches on. XBRL is
# reported here as plain ``XML``; discriminating it from other XML is a later
# content step. The image types are the same set Docling's vision pipeline
# accepts, so detection and the parser agree on what counts as an image.
# Anything absent maps to ``UNKNOWN`` and routes to quarantine.
_MEDIA_TYPE_TO_FORMAT: dict[str, DetectedFormat] = {
    "application/pdf": DetectedFormat.PDF,
    "image/png": DetectedFormat.IMAGE,
    "image/jpeg": DetectedFormat.IMAGE,
    "image/tiff": DetectedFormat.IMAGE,
    "image/bmp": DetectedFormat.IMAGE,
    "image/webp": DetectedFormat.IMAGE,
    "image/gif": DetectedFormat.IMAGE,
    "text/html": DetectedFormat.HTML,
    "application/xhtml+xml": DetectedFormat.HTML,
    "application/xml": DetectedFormat.XML,
    "text/xml": DetectedFormat.XML,
    "text/plain": DetectedFormat.TEXT,
}


@registry.register("format_detector", "tika")
class TikaFormatDetector:
    """Detects a document's format by asking a Tika server's ``/detect`` endpoint."""

    def __init__(
        self,
        url: str,
        timeout_s: float = 10.0,
        prefix_bytes: int = _DETECT_PREFIX_BYTES,
    ) -> None:
        """Create a detector bound to a Tika server.

        Args:
          url: Base URL of the Tika server, e.g. ``http://finrag-tika:9998``.
          timeout_s: Per-request timeout in seconds.
          prefix_bytes: Number of leading bytes to send for detection.
        """
        self._detect_url = url.rstrip("/") + "/detect/stream"
        self._prefix_bytes = prefix_bytes
        self._client = httpx.Client(timeout=timeout_s)

    def detect(self, data: bytes) -> DetectedFormat:
        """Return the coarse format of the given document bytes.

        Args:
          data: The document's bytes; only the leading prefix is inspected.

        Returns:
          The detected format, or :attr:`DetectedFormat.UNKNOWN` if the media
          type is one the router does not handle.

        Raises:
          IngestionError: The Tika server could not be reached or errored.
        """
        media_type = self._detect_media_type(data[: self._prefix_bytes])
        return _MEDIA_TYPE_TO_FORMAT.get(media_type, DetectedFormat.UNKNOWN)

    def _detect_media_type(self, head: bytes) -> str:
        """Ask Tika for the media type of a byte prefix, normalized to the bare type.

        Tika may append parameters (e.g. ``; charset=UTF-8``); they are dropped
        so the result is a plain, lowercased media type for table lookup.
        """
        try:
            response = self._client.put(
                self._detect_url,
                content=head,
                headers={"Accept": "text/plain"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise IngestionError(f"tika detect failed: {exc}") from exc
        return response.text.split(";", 1)[0].strip().lower()
