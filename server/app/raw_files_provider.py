"""get_raw_file_store — a process-wide `ragcore.raw_files.S3RawFileStore`
configured entirely from `app.settings.Settings`, used by both the
`api` container (uploading a freshly-received file before enqueuing
its ingestion job) and the `worker` container (downloading that same
file by `source_id` before calling `Corpus.ingest()`) — the MinIO
handoff mechanism ADR 0014 decided on, reusing `S3RawFileStore` rather
than inventing a second object-storage abstraction.
"""

from __future__ import annotations

from functools import lru_cache

from app.settings import get_settings
from ragcore.raw_files.s3 import S3RawFileStore


@lru_cache(maxsize=1)
def get_raw_file_store() -> S3RawFileStore:
    settings = get_settings()
    if not settings.s3_endpoint_url:
        raise RuntimeError(
            "S3_ENDPOINT_URL (and S3_BUCKET/S3_ACCESS_KEY_ID/S3_SECRET_ACCESS_KEY) "
            "must be set to use object storage (async ingestion, backups)"
        )
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
    )
    return S3RawFileStore(settings.s3_bucket, prefix=settings.s3_raw_files_prefix, client=client)
