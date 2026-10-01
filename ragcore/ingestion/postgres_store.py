"""IngestCheckpointStore implementation backed by Postgres (via psycopg
3) — proves ragcore.ingestion.IngestCheckpointStore is a genuine
swappable protocol, not SQLite-shaped in practice, and is the Postgres
half of docs/ARCHITECTURE.md Phase 2 gap list item 12 ("Postgres for
metadata, jobs...").

Schema and method contracts mirror SqliteIngestCheckpointStore exactly
(see its module docstring for the checkpoint-ordering rationale, which
applies here unchanged) — Postgres's `ON CONFLICT ... DO UPDATE SET`
syntax is the same upsert pattern SQLite already uses throughout this
codebase, so the SQL itself barely changes; the real differences are
`%s` placeholders (not `?`), a real `TIMESTAMPTZ` column (not a TEXT
isoformat string), and psycopg's `connection.transaction()` context
manager in place of explicit `BEGIN IMMEDIATE`/`COMMIT`/`ROLLBACK`.
`backup_to`/`restore_from_backup` (SQLite-file-specific, not part of the
`IngestCheckpointStore` protocol) have no equivalent here — Postgres has
its own backup tooling (`pg_dump`/WAL archiving) outside this library's
scope.

**Tested only against a real Postgres, opt-in** (`pytest -m postgres`,
`RAGCORE_TEST_POSTGRES_URL` — see tests/test_postgres_ingest_checkpoint_store.py):
unlike every other default store in this codebase, there is no embedded,
in-process Postgres to test against unconditionally the way SQLite's
stdlib module allows. Written and reviewed against psycopg 3's documented
API and standard PostgreSQL syntax, but — unlike the SQLite stores, whose
tests run and pass in every environment by default — treat this as
unverified by actual execution until it has been run against a live
Postgres server at least once in your own environment.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from ragcore.errors import StoreError, wrap_backend_errors
from ragcore.ingestion.base import DeadLetter

try:
    import psycopg
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    psycopg = None
else:
    _IMPORT_ERROR = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_group_runs (
    source_id TEXT NOT NULL,
    group_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (source_id, group_id)
);

CREATE TABLE IF NOT EXISTS ingest_dead_letters (
    path TEXT PRIMARY KEY,
    error TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PostgresIngestCheckpointStore:
    """Postgres-backed IngestCheckpointStore. `conninfo` is a standard
    libpq connection string (e.g. `postgresql://user:pass@host/db`);
    `client` lets a caller (or a test) inject an already-open
    `psycopg.Connection` instead."""

    def __init__(self, conninfo: str = "", *, client: Any | None = None) -> None:
        if client is None and psycopg is None:
            raise StoreError(
                "Postgres ingest checkpoint storage requires the 'postgres' extra: "
                "install ragcore[postgres]"
            ) from _IMPORT_ERROR

        self._lock = threading.Lock()
        self._conn = client or psycopg.connect(conninfo, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute(_SCHEMA)

    @wrap_backend_errors
    def checkpoint(self, source_id: str, group_id: str) -> str | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT fingerprint FROM ingest_group_runs "
                "WHERE source_id = %s AND group_id = %s AND status = 'done'",
                (source_id, group_id),
            )
            row = cur.fetchone()
        return row[0] if row else None

    @wrap_backend_errors
    def mark_group_done(self, *, source_id: str, group_id: str, fingerprint: str) -> None:
        with self._lock, self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ingest_group_runs "
                "(source_id, group_id, fingerprint, status, error, updated_at) "
                "VALUES (%s, %s, %s, 'done', NULL, %s) "
                "ON CONFLICT (source_id, group_id) DO UPDATE SET "
                "fingerprint = excluded.fingerprint, status = 'done', error = NULL, "
                "updated_at = excluded.updated_at",
                (source_id, group_id, fingerprint, _now()),
            )

    @wrap_backend_errors
    def mark_group_failed(
        self, *, source_id: str, group_id: str, fingerprint: str, error: str
    ) -> None:
        with self._lock, self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ingest_group_runs "
                "(source_id, group_id, fingerprint, status, error, updated_at) "
                "VALUES (%s, %s, %s, 'failed', %s, %s) "
                "ON CONFLICT (source_id, group_id) DO UPDATE SET "
                "fingerprint = excluded.fingerprint, status = 'failed', "
                "error = excluded.error, updated_at = excluded.updated_at",
                (source_id, group_id, fingerprint, error, _now()),
            )

    @wrap_backend_errors
    def remove_source(self, source_id: str) -> None:
        with self._lock, self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute("DELETE FROM ingest_group_runs WHERE source_id = %s", (source_id,))

    @wrap_backend_errors
    def mark_dead_letter(self, *, path: str, error: str, attempts: int) -> None:
        with self._lock, self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ingest_dead_letters (path, error, attempts, updated_at) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (path) DO UPDATE SET "
                "error = excluded.error, attempts = excluded.attempts, "
                "updated_at = excluded.updated_at",
                (path, error, attempts, _now()),
            )

    @wrap_backend_errors
    def dead_letters(self) -> list[DeadLetter]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT path, error, attempts, updated_at FROM ingest_dead_letters "
                "ORDER BY updated_at"
            )
            rows = cur.fetchall()
        return [
            DeadLetter(path=r[0], error=r[1], attempts=r[2], updated_at=r[3].isoformat())
            for r in rows
        ]

    @wrap_backend_errors
    def clear_dead_letter(self, path: str) -> None:
        with self._lock, self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute("DELETE FROM ingest_dead_letters WHERE path = %s", (path,))
