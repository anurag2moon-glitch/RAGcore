"""Restore a backup written by scripts/backup.py - the exact inverse
of each of its three steps, reading from the same S3/MinIO bucket's
backups prefix (app.naming.backup_object_key). Meant to run inside the
api/worker image, same as backup.py:

    docker compose exec api python scripts/restore.py <tag>

Restoring Postgres truncates each registry table before loading the
backup's rows back in (COPY ... FORMAT BINARY) - a restore is meant to
replace current state with the backup's state, not merge the two.
Restoring the data dir/raw-files bucket extracts over whatever is
there, which is the correct "wipe and replace" semantics for the exact
reason a restore script tests: this must result in the backed-up
state, not a merge with whatever was already present. Run it against
a throwaway stack when testing - see docs/DEPLOYMENT.md's restore
section for the tested procedure.
"""

from __future__ import annotations

import io
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

from app.naming import API_KEYS_TABLE, TENANTS_TABLE, backup_object_key
from app.raw_files_provider import get_s3_client
from app.settings import get_settings
from scripts.backup import data_dir_roles


def _restore_postgres(tag: str) -> None:
    import psycopg
    from botocore.exceptions import ClientError

    settings = get_settings()
    if not settings.database_url:
        print("DATABASE_URL not set - skipping Postgres restore", file=sys.stderr)
        return

    client = get_s3_client()
    with psycopg.connect(settings.database_url) as conn:
        for table in (TENANTS_TABLE, API_KEYS_TABLE):
            key = backup_object_key(tag, f"postgres.{table}.bin")
            buf = io.BytesIO()
            try:
                client.download_fileobj(settings.s3_bucket, key, buf)
            except ClientError as e:
                print(f"no backup found for table {table!r} ({key}): {e} - skipping", file=sys.stderr)
                continue
            buf.seek(0)

            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {table}")
                with cur.copy(f"COPY {table} FROM STDIN (FORMAT BINARY)") as copy:
                    while chunk := buf.read(65536):
                        copy.write(chunk)
            conn.commit()
            print(f"restored Postgres table {table!r} from s3://{settings.s3_bucket}/{key}")


def _restore_data_dir(tag: str) -> None:
    settings = get_settings()
    client = get_s3_client()
    key = backup_object_key(tag, "data.tar.gz")

    from botocore.exceptions import ClientError

    buf = io.BytesIO()
    try:
        client.download_fileobj(settings.s3_bucket, key, buf)
    except ClientError as e:
        print(f"no data-dir backup found ({key}): {e} - skipping", file=sys.stderr)
        return
    buf.seek(0)

    roles = data_dir_roles()
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            tar.extractall(path=tmp, filter="data")

        archived_roles = {p.name for p in tmp.iterdir()}
        for role in archived_roles & roles.keys():
            dest = roles[role]
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp / role), str(dest))
            print(f"restored {role} -> {dest}")


def _restore_raw_files(tag: str) -> None:
    settings = get_settings()
    client = get_s3_client()
    key = backup_object_key(tag, "raw-files.tar.gz")

    from botocore.exceptions import ClientError

    buf = io.BytesIO()
    try:
        client.download_fileobj(settings.s3_bucket, key, buf)
    except ClientError as e:
        print(f"no raw-files backup found ({key}): {e} - skipping", file=sys.stderr)
        return
    buf.seek(0)

    count = 0
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        for member in tar.getmembers():
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            client.put_object(Bucket=settings.s3_bucket, Key=member.name, Body=extracted.read())
            count += 1
    print(f"restored {count} raw file(s) from s3://{settings.s3_bucket}/{key}")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python scripts/restore.py <tag>", file=sys.stderr)
        sys.exit(1)
    tag = sys.argv[1]

    _restore_postgres(tag)
    _restore_data_dir(tag)
    _restore_raw_files(tag)
    print(f"restore {tag!r} complete")


if __name__ == "__main__":
    main()
