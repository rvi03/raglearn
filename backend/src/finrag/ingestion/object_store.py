"""Object I/O against the MinIO store.

Isolates the network side effects the pipeline needs: pull an object's bytes
given a bucket and key (the consumer's read path), and write an uploaded file's
bytes under a key (the upload endpoint's write path — each write raises the
bucket notification that drives ingestion). The MinIO SDK is synchronous, so the
blocking calls run in a worker thread to keep the caller's event loop free.
"""

from __future__ import annotations

import asyncio
import io
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error

from finrag.core.errors import IngestionError


class ObjectStore:
    """Reads and writes objects in a MinIO bucket."""

    def __init__(self, endpoint: str, access_key: str, secret_key: str) -> None:
        """Create a store client.

        Args:
          endpoint: MinIO endpoint URL, e.g. ``http://localhost:9000``. The
            scheme selects TLS; the host:port is passed to the SDK.
          access_key: MinIO access key.
          secret_key: MinIO secret key.
        """
        parsed = urlparse(endpoint)
        self._client = Minio(
            parsed.netloc,
            access_key=access_key,
            secret_key=secret_key,
            secure=parsed.scheme == "https",
        )

    def fetch_sync(self, bucket: str, key: str) -> bytes:
        """Read an object's full body, releasing the connection afterward.

        Synchronous; for callers already off the event loop (e.g. XBRL bundle
        assembly inside the router's worker thread).

        Args:
          bucket: Bucket name.
          key: Object key.

        Returns:
          The object's raw bytes.

        Raises:
          IngestionError: The object could not be read.
        """
        try:
            response = self._client.get_object(bucket, key)
        except S3Error as exc:
            raise IngestionError(f"failed to fetch {bucket}/{key}: {exc}") from exc
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def list_prefix(self, bucket: str, prefix: str) -> list[str]:
        """List the keys of all objects under a prefix, recursively.

        Used to gather the sibling files of an XBRL filing (instance, schema,
        linkbases) that share its accession folder.

        Args:
          bucket: Bucket name.
          prefix: Key prefix, e.g. ``us/apple/0000320193-24-000123-xbrl/``.

        Returns:
          The object keys under the prefix.

        Raises:
          IngestionError: The listing could not be read.
        """
        try:
            return [
                obj.object_name
                for obj in self._client.list_objects(bucket, prefix=prefix, recursive=True)
                if obj.object_name is not None
            ]
        except S3Error as exc:
            raise IngestionError(f"failed to list {bucket}/{prefix}: {exc}") from exc

    async def fetch(self, bucket: str, key: str) -> bytes:
        """Return the bytes of an object.

        Args:
          bucket: Bucket name.
          key: Object key (already URL-decoded).

        Returns:
          The object's raw bytes.

        Raises:
          IngestionError: The object could not be read.
        """
        return await asyncio.to_thread(self.fetch_sync, bucket, key)

    def put_sync(
        self, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> None:
        """Write an object's bytes under a key, overwriting any existing object.

        Synchronous; the upload endpoint calls the async :meth:`put` wrapper. The
        bucket must already exist with its put-notification configured (infra
        owns that) — creating it here would yield a bucket that does not trigger
        ingestion.

        Args:
          bucket: Bucket name.
          key: Object key, e.g. ``us/<upload_id>/aapl-10k/aapl-20240928.htm``.
          data: Object body.
          content_type: MIME type recorded on the object.

        Raises:
          IngestionError: The object could not be written.
        """
        try:
            self._client.put_object(
                bucket, key, io.BytesIO(data), length=len(data), content_type=content_type
            )
        except S3Error as exc:
            raise IngestionError(f"failed to put {bucket}/{key}: {exc}") from exc

    async def put(
        self, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> None:
        """Write an object's bytes under a key.

        Args:
          bucket: Bucket name.
          key: Object key.
          data: Object body.
          content_type: MIME type recorded on the object.

        Raises:
          IngestionError: The object could not be written.
        """
        await asyncio.to_thread(self.put_sync, bucket, key, data, content_type)
