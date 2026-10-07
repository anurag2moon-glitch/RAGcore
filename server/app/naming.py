"""The one place table names, object storage key prefixes, and local
tenant data paths are decided — "Tenant data paths, object storage
prefixes and table naming must come from config and a single helper
module, not hardcoded, so the tenant isolation design can change later
without touching deployment" (the minimal-staging-deployment request's
own rule). Every one of these is read from `app.settings.Settings`
where it's configurable at all; the two Postgres table names are not
currently settings-configurable (changing a table name is a schema
migration, not a runtime config choice), but still live here, not
inlined at each of their three call sites (the Alembic migration,
`PostgresApiKeyStore`, and the seed script).
"""

from __future__ import annotations

from pathlib import Path

from app.settings import get_settings

TENANTS_TABLE = "tenants"
API_KEYS_TABLE = "api_keys"


def tenant_data_path(tenant_id: str | None) -> Path:
    """A local filesystem directory scoped to one tenant (or, for
    `None`, a shared scratch area with no tenant of its own) under
    `TENANT_DATA_ROOT`. Today's one real caller is the ingestion
    worker's download-before-ingest step
    (`app.worker.ingest_job.run_ingest_job`) — a file is staged here,
    not in a bare system temp directory, so that if the tenant
    isolation design later needs per-tenant filesystem boundaries
    (e.g. a dedicated volume or mount per tenant), this is the one
    place that changes."""
    root = Path(get_settings().tenant_data_root)
    return root / (tenant_id if tenant_id is not None else "_shared")


def raw_file_object_key(source_id: str) -> str:
    """The S3/MinIO object key for a raw ingested file's original
    bytes, under the configured prefix — matches the key shape
    `ragcore.raw_files.S3RawFileStore` itself builds internally
    (`f"{prefix}{source_id}"`), exposed here so a caller enqueuing a
    job can reference the same key before/without constructing a full
    `S3RawFileStore`."""
    return f"{get_settings().s3_raw_files_prefix}{source_id}"


def backup_object_key(tag: str, name: str) -> str:
    """The S3/MinIO object key for one file within one named backup
    (e.g. a Postgres dump, a LanceDB/SQLite snapshot archive) under
    the configured backups prefix."""
    return f"{get_settings().s3_backups_prefix}{tag}/{name}"
