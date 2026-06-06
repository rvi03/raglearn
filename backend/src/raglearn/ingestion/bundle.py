"""XBRL bundle assembly from the object store.

An XBRL filing is not one file: the inline-XBRL instance needs its schema
(``.xsd``) and linkbases (``_cal``/``_def``/``_lab``/``_pre``) to resolve facts
to concepts. The event pipeline delivers one object per notification, so when
the instance is seen the rest of the bundle is pulled from its accession folder
in the store and materialized side by side in a temp directory, which is where
Arelle expects to resolve the references from.

Only the instance triggers assembly; lone linkbases and schemas are recognized
as bundle members and skipped upstream (see :mod:`raglearn.ingestion.xbrl_extract`).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from raglearn.core.errors import IngestionError
from raglearn.core.types import RawDocument
from raglearn.ingestion.object_store import ObjectStore

logger = logging.getLogger(__name__)

# Extensions that make up an XBRL bundle. Exhibits and images in the same folder
# are not referenced by the instance, so they are left unfetched.
_BUNDLE_EXTENSIONS = frozenset({".htm", ".html", ".xml", ".xsd"})


def _accession_stem(key: str) -> str:
    """Return the filename stem shared by a filing's bundle files.

    SEC names a filing's instance, schema, and linkbases with a common stem
    (e.g. ``aapl-20240928``); matching on it gathers the bundle while skipping
    co-located exhibits.
    """
    return Path(key).name.split(".", 1)[0]


class BundleAssembler:
    """Materializes an XBRL filing's bundle from the object store to local disk."""

    def __init__(self, store: ObjectStore) -> None:
        """Bind the assembler to an object store.

        Args:
          store: Source of the bundle's object bytes.
        """
        self._store = store

    @contextmanager
    def materialize(self, document: RawDocument) -> Iterator[Path | None]:
        """Fetch a *complete* filing bundle to a temp dir and yield its instance path.

        A filing's numeric facts need only the instance plus its schema
        (``.xsd``); the linkbases add labels and ordering, not facts. So the
        bundle is treated as complete once both the instance and the schema are
        present in the store. The event pipeline delivers files one at a time and
        in any order, so when the instance arrives before its schema (or vice
        versa) this yields ``None`` — the caller defers, and the later sibling
        event retries, the last of the two to land completing the bundle.

        Args:
          document: The triggering document (instance or schema); its
            ``source_bucket`` and ``doc_id`` locate the bundle by accession.

        Yields:
          The instance's path inside the temp directory once the bundle is
          complete, or ``None`` if it is not yet (no temp dir is created then).

        Raises:
          IngestionError: The document lacks a source bucket, or the bundle
            could not be read.
        """
        if document.source_bucket is None:
            raise IngestionError(
                f"cannot assemble XBRL bundle without a source bucket: {document.doc_id}"
            )
        bucket = document.source_bucket
        key = document.doc_id
        prefix = key.rsplit("/", 1)[0] + "/" if "/" in key else ""
        stem = _accession_stem(key)
        present = set(self._store.list_prefix(bucket, prefix))

        instance_key = self._find_instance(prefix, stem, present)
        schema_key = f"{prefix}{stem}.xsd"
        if instance_key is None or schema_key not in present:
            missing = "instance" if instance_key is None else "schema (.xsd)"
            logger.info("XBRL bundle %s incomplete (no %s yet); deferring", prefix or stem, missing)
            yield None
            return

        siblings = [
            sib
            for sib in present
            if Path(sib).name.startswith(stem) and Path(sib).suffix.lower() in _BUNDLE_EXTENSIONS
        ]
        tmp = Path(tempfile.mkdtemp(prefix="raglearn-xbrl-"))
        try:
            for sib in siblings:
                (tmp / Path(sib).name).write_bytes(self._store.fetch_sync(bucket, sib))
            logger.info("assembled %d bundle file(s) for %s", len(siblings), instance_key)
            yield tmp / Path(instance_key).name
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def _find_instance(prefix: str, stem: str, present: set[str]) -> str | None:
        """Return the bundle's instance key, preferring the inline-XBRL ``.htm``.

        The instance shares the stem with the schema; linkbases carry a suffix
        (``_lab`` etc.), so a bare ``{stem}.{ext}`` is the instance, not a member.
        """
        for ext in (".htm", ".html", ".xml"):
            candidate = f"{prefix}{stem}{ext}"
            if candidate in present:
                return candidate
        return None
