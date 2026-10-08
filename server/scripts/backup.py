"""Backup Postgres (the tenant/api-key registry), the /data volume
(LanceDB + the rules SQLite file + tenant data), and object storage's
raw-files bucket, bundled under one named `tag` and uploaded to the
SAME S3/MinIO bucket's own backups prefix (app.naming.backup_object_key)
rather than a separate host-mounted directory - this script only needs
network access to Postgres/S3 (both already reachable wherever it
runs) and read access to the mounted data volume it runs alongside.
Meant to run inside the api/worker image, which has both:

    docker compose exec api python scripts/backup.py <tag>

Postgres is dumped table-by-table via psycopg's COPY ... (FORMAT BINARY)
- reusing the driver already in this image rather than needing the
`pg_dump` binary installed (python:3.12-slim doesn't ship it, and this
server's Postgres schema is two small registry tables, not a full
application database `pg_dump` is designed for). The /data volume and
the raw-files bucket are each bundled as one tar, uploaded as one
object - see restore.py for the exact inverse of every step here.
"""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

from app.naming import API_KEYS_TABLE, TENANTS_TABLE, backup_object_key
from app.raw_files_provider import get_s3_client
from app.settings import get_settings


def _backup_postgres(tag: str) -> None:
    import psycopg

    settings = get_settings()
    if not settings.database_url:
        print("DATABASE_URL not set - skipping Postgres backup", file=sys.stderr)
        return

    client = get_s3_client()
    with psycopg.connect(settings.database_url) as conn:
        for table in (TENANTS_TABLE, API_KEYS_TABLE):
            buf = io.BytesIO()
            with conn.cursor() as cur, cur.copy(f"COPY {table} TO STDOUT (FORMAT BINARY)") as copy:
                for chunk in copy:
                    buf.write(bytes(chunk))
            buf.seek(0)
            key = backup_object_key(tag, f"postgres.{table}.bin")
            client.upload_fileobj(buf, settings.s3_bucket, key)
            print(f"backed up Postgres table {table!r} -> s3://{settings.s3_bucket}/{key}")


def data_dir_roles() -> dict[str, Path]:
    """Fixed, semantic archive names for each configured local data
    path - not the path's own basename, since that's deployment-
    specific (and could in principle differ between the environment a
    backup was taken in and the one it's restored into). restore.py
    maps these same fixed names back onto whatever paths the *current*
    settings say, rather than guessing from basenames in the archive.
    """
    settings = get_settings()
    roles: dict[str, Path] = {}
    if settings.ragcore_db_path:
        roles["lancedb"] = Path(settings.ragcore_db_path)
    if settings.ragcore_rules_db_path:
        roles["rules_sqlite"] = Path(settings.ragcore_rules_db_path)
    roles["tenant_data"] = Path(settings.tenant_data_root)
    return roles


def _backup_data_dir(tag: str) -> None:
    settings = get_settings()
    existing = {role: path for role, path in data_dir_roles().items() if path.exists()}
    if not existing:
        print("no local data paths exist - skipping data-dir backup", file=sys.stderr)
        return

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for role, path in existing.items():
            tar.add(path, arcname=role)
    buf.seek(0)

    client = get_s3_client()
    key = backup_object_key(tag, "data.tar.gz")
    client.upload_fileobj(buf, settings.s3_bucket, key)
    print(f"backed up data dir ({list(existing)}) -> s3://{settings.s3_bucket}/{key}")


def _backup_raw_files(tag: str) -> None:
    settings = get_settings()
    client = get_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    keys = [
        obj["Key"]
        for page in paginator.paginate(Bucket=settings.s3_bucket, Prefix=settings.s3_raw_files_prefix)
        for obj in page.get("Contents", [])
    ]
    if not keys:
        print("no raw files to back up", file=sys.stderr)
        return

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for key in keys:
            obj = client.get_object(Bucket=settings.s3_bucket, Key=key)
            data = obj["Body"].read()
            info = tarfile.TarInfo(name=key)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)

    dest_key = backup_object_key(tag, "raw-files.tar.gz")
    client.upload_fileobj(buf, settings.s3_bucket, dest_key)
    print(f"backed up {len(keys)} raw file(s) -> s3://{settings.s3_bucket}/{dest_key}")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python scripts/backup.py <tag>", file=sys.stderr)
        sys.exit(1)
    tag = sys.argv[1]

    _backup_postgres(tag)
    _backup_data_dir(tag)
    _backup_raw_files(tag)
    print(f"backup {tag!r} complete")


if __name__ == "__main__":
    main()
