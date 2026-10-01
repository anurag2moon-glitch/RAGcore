"""RawFileStore implementation backed by any S3-compatible object store
(AWS S3 itself, or a compatible service — MinIO, R2, B2, ... — via
`endpoint_url`).

Object key is `f"{prefix}{source_id}"`, with no suffix: the original
filename (and therefore its extension) is preserved separately as S3
object metadata, not encoded into the key. This keeps `get()`/`exists()`/
`delete()` a single direct key lookup (no listing needed to find "the
object for this source_id" regardless of its original extension), and
`original_filename()` lets a caller recover the right extension for
`dest` before downloading, if it wants one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragcore.errors import StoreError, wrap_backend_errors

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    boto3 = None
    ClientError = Exception  # type: ignore[assignment,misc]
else:
    _IMPORT_ERROR = None

_NOT_FOUND_CODES = {"404", "NoSuchKey"}


def _is_not_found(e: ClientError) -> bool:
    return e.response.get("Error", {}).get("Code") in _NOT_FOUND_CODES


class S3RawFileStore:
    """Default RawFileStore: one S3-compatible bucket, objects keyed by
    source_id with the original filename preserved as object metadata."""

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        client: Any | None = None,
        endpoint_url: str | None = None,
        region_name: str | None = None,
    ) -> None:
        if client is None and boto3 is None:
            raise StoreError(
                "S3 raw file storage requires the 's3' extra: install ragcore[s3]"
            ) from _IMPORT_ERROR

        self._bucket = bucket
        self._prefix = prefix
        self._client = client or boto3.client(
            "s3", endpoint_url=endpoint_url, region_name=region_name
        )

    def _key(self, source_id: str) -> str:
        return f"{self._prefix}{source_id}"

    @wrap_backend_errors
    def put(self, source_id: str, path: Path) -> str:
        key = self._key(source_id)
        self._client.upload_file(
            str(path), self._bucket, key, ExtraArgs={"Metadata": {"original-filename": path.name}}
        )
        return f"s3://{self._bucket}/{key}"

    @wrap_backend_errors
    def get(self, source_id: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(self._bucket, self._key(source_id), str(dest))
        except ClientError as e:
            if _is_not_found(e):
                raise StoreError(f"no raw file stored under source_id={source_id!r}") from e
            raise

    @wrap_backend_errors
    def exists(self, source_id: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=self._key(source_id))
            return True
        except ClientError as e:
            if _is_not_found(e):
                return False
            raise

    @wrap_backend_errors
    def delete(self, source_id: str) -> bool:
        if not self.exists(source_id):
            return False
        self._client.delete_object(Bucket=self._bucket, Key=self._key(source_id))
        return True

    @wrap_backend_errors
    def original_filename(self, source_id: str) -> str | None:
        try:
            response = self._client.head_object(Bucket=self._bucket, Key=self._key(source_id))
        except ClientError as e:
            if _is_not_found(e):
                return None
            raise
        return response.get("Metadata", {}).get("original-filename")
