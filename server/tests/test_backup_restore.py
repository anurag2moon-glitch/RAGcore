"""scripts/backup.py / scripts/restore.py, exercised for real against a
moto-mocked S3/MinIO bucket and real local temp directories - the
"LanceDB data + tenant data" and "object storage" halves of "Backup
script and restore script ... with a tested restore" get genuine,
unconditional round-trip coverage here. The Postgres half is marked
`postgres` (same opt-in, skips-cleanly precedent as every other
Postgres-touching test in this server - ADR 0005/0016) since no live
Postgres server is available in this environment; its functions are
still imported and exercised for import/syntax correctness by every
other test in this file running at all.
"""

from __future__ import annotations

import os

import boto3
import pytest
from app.raw_files_provider import get_s3_client
from app.settings import Settings
from moto import mock_aws

from scripts import backup, restore

BUCKET = "ragcore-test-bucket"


@pytest.fixture
def s3_backup_env(tmp_path, monkeypatch):
    lancedb_path = tmp_path / "data" / "corpus.lancedb"
    lancedb_path.mkdir(parents=True)
    (lancedb_path / "table.lance").write_text("fake lancedb content")

    rules_path = tmp_path / "data" / "rules.sqlite3"
    rules_path.write_text("fake sqlite content")

    tenant_root = tmp_path / "data" / "tenants"
    (tenant_root / "staging").mkdir(parents=True)
    (tenant_root / "staging" / "incoming.txt").write_text("fake tenant file")

    settings = Settings(
        s3_endpoint_url="http://irrelevant:9000",
        s3_bucket=BUCKET,
        s3_raw_files_prefix="raw-files/",
        s3_backups_prefix="backups/",
        ragcore_db_path=str(lancedb_path),
        ragcore_rules_db_path=str(rules_path),
        tenant_data_root=str(tenant_root),
    )

    with mock_aws():
        # moto only intercepts calls to AWS's own default endpoints, not
        # a custom endpoint_url (the MinIO-style config this server's
        # Settings normally requires) - so the test client omits
        # endpoint_url, and scripts.backup/restore's own get_s3_client
        # is monkeypatched directly to return it, the same pattern
        # conftest.py's _async_ingest_infra already uses for the same
        # reason.
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        client.put_object(Bucket=BUCKET, Key="raw-files/abc123", Body=b"a real uploaded book's bytes")

        get_s3_client.cache_clear()
        monkeypatch.setattr("scripts.backup.get_settings", lambda: settings)
        monkeypatch.setattr("scripts.restore.get_settings", lambda: settings)
        monkeypatch.setattr("scripts.backup.get_s3_client", lambda: client)
        monkeypatch.setattr("scripts.restore.get_s3_client", lambda: client)

        yield settings, client, tenant_root, lancedb_path, rules_path

    get_s3_client.cache_clear()


def test_backup_then_restore_round_trips_the_data_dir(s3_backup_env, tmp_path) -> None:
    _settings, _client, tenant_root, lancedb_path, rules_path = s3_backup_env

    backup._backup_data_dir("tag1")

    # Simulate total data loss, then restore into the exact same configured paths.
    import shutil

    shutil.rmtree(lancedb_path)
    rules_path.unlink()
    shutil.rmtree(tenant_root)

    restore._restore_data_dir("tag1")

    assert (lancedb_path / "table.lance").read_text() == "fake lancedb content"
    assert rules_path.read_text() == "fake sqlite content"
    assert (tenant_root / "staging" / "incoming.txt").read_text() == "fake tenant file"


def test_backup_then_restore_round_trips_raw_files(s3_backup_env) -> None:
    _settings, client, *_ = s3_backup_env

    backup._backup_raw_files("tag1")

    # Simulate the raw-files bucket being wiped, then restore it.
    client.delete_object(Bucket=BUCKET, Key="raw-files/abc123")
    assert "Contents" not in client.list_objects_v2(Bucket=BUCKET, Prefix="raw-files/")

    restore._restore_raw_files("tag1")

    restored = client.get_object(Bucket=BUCKET, Key="raw-files/abc123")
    assert restored["Body"].read() == b"a real uploaded book's bytes"


def test_restore_with_no_backup_present_skips_cleanly_without_raising(s3_backup_env) -> None:
    restore._restore_data_dir("never-backed-up")
    restore._restore_raw_files("never-backed-up")


def test_data_dir_roles_omits_rules_sqlite_when_unconfigured(s3_backup_env, monkeypatch) -> None:
    settings, *_ = s3_backup_env
    no_rules_settings = settings.model_copy(update={"ragcore_rules_db_path": None})
    monkeypatch.setattr("scripts.backup.get_settings", lambda: no_rules_settings)

    roles = backup.data_dir_roles()

    assert "rules_sqlite" not in roles
    assert "lancedb" in roles
    assert "tenant_data" in roles


@pytest.mark.postgres
def test_backup_then_restore_round_trips_postgres_tables() -> None:
    """Opt-in, against a real Postgres server - same precedent as
    every other `postgres`-marked test in this server (ADR 0005/0016).
    Run with RAGCORE_TEST_POSTGRES_URL set and `pytest -m postgres`."""
    import psycopg
    from app.naming import API_KEYS_TABLE, TENANTS_TABLE

    database_url = os.environ["RAGCORE_TEST_POSTGRES_URL"]
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {TENANTS_TABLE} (tenant_id, display_name) "
            "VALUES ('backup-test-tenant', 'Backup Test') "
            "ON CONFLICT (tenant_id) DO NOTHING"
        )
        cur.execute(
            f"INSERT INTO {API_KEYS_TABLE} (api_key, tenant_id) "
            "VALUES ('backup-test-key', 'backup-test-tenant') "
            "ON CONFLICT (api_key) DO NOTHING"
        )
        conn.commit()

    backup._backup_postgres("pg-tag")

    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(f"DELETE FROM {API_KEYS_TABLE} WHERE api_key = 'backup-test-key'")
        cur.execute(f"DELETE FROM {TENANTS_TABLE} WHERE tenant_id = 'backup-test-tenant'")
        conn.commit()

    restore._restore_postgres("pg-tag")

    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT tenant_id FROM {TENANTS_TABLE} WHERE tenant_id = 'backup-test-tenant'")
        assert cur.fetchone() is not None
