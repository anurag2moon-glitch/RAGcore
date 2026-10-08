"""get_raw_file_store/get_s3_client — object storage access configured
entirely from `app.settings.Settings`. `get_raw_file_store` (a process-
wide `ragcore.raw_files.S3RawFileStore`) is used by both the `api`
container (uploading a freshly-received file before enqueuing its
ingestion job) and the `worker` container (downloading that same file
by `source_id` before calling `Corpus.ingest()`) — the MinIO handoff
mechanism ADR 0014 decided on, reusing `S3RawFileStore` rather than
inventing a second object-storage abstraction. `get_s3_client` is the
one raw `boto3` client underneath it, exposed separately for
`scripts/backup.py`/`scripts/restore.py`, which need operations
(`list_objects_v2`, arbitrary keys under `s3_backups_prefix`)
`S3RawFileStore`'s source_id-keyed interface doesn't cover.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.settings import get_settings
from ragcore.raw_files.s3 import S3RawFileStore


def _require_s3_settings() -> None:
    if not get_settings().s3_endpoint_url:
        raise RuntimeError(
            "S3_ENDPOINT_URL (and S3_BUCKET/S3_ACCESS_KEY_ID/S3_SECRET_ACCESS_KEY) "
            "must be set to use object storage (async ingestion, backups)"
        )


@lru_cache(maxsize=1)
def get_s3_client() -> Any:
    _require_s3_settings()
    settings = get_settings()
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
    )


@lru_cache(maxsize=1)
def get_raw_file_store() -> S3RawFileStore:
    settings = get_settings()
    _require_s3_settings()
    return S3RawFileStore(settings.s3_bucket, prefix=settings.s3_raw_files_prefix, client=get_s3_client())
