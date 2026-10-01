"""IngestCheckpointStore implementation backed by stdlib sqlite3 — own
file, own connection, copying ragcore.contextual.sqlite_store.SqliteContextStore's
connection setup and transaction pattern exactly.

Checkpoint ordering: Corpus.ingest() upserts a group's chunks into the
VectorStore *before* calling mark_group_done() for it (see
ragcore/corpus.py). A crash between the two leaves the chunks stored but
the group not yet checkpointed, so a retry simply re-embeds and re-upserts
that one group — wasted work (VectorStore.upsert is idempotent by
chunk_id), never a correctness gap. The reverse ordering (checkpoint, then
upsert) would risk the opposite: a checkpoint claiming a group is done
while its chunks are still missing from the store, which a resumed
ingest() would then skip re-embedding — silent data loss. This file does
not enforce that ordering itself (it has no visibility into the
VectorStore); Corpus.ingest() is what's responsible for calling
mark_group_done() only after a successful upsert.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from ragcore.ingestion.base import DeadLetter


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_group_runs (
    source_id TEXT NOT NULL,
    group_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (source_id, group_id)
);

CREATE TABLE IF NOT EXISTS ingest_dead_letters (
    path TEXT PRIMARY KEY,
    error TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class SqliteIngestCheckpointStore:
    """Default IngestCheckpointStore: one SQLite file, one row per
    (source_id, group_id) attempt, plus a dead-letter table for
    ingest_many()."""

    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)

    def checkpoint(self, source_id: str, group_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT fingerprint FROM ingest_group_runs "
            "WHERE source_id = ? AND group_id = ? AND status = 'done'",
            (source_id, group_id),
        ).fetchone()
        return row[0] if row else None

    def mark_group_done(self, *, source_id: str, group_id: str, fingerprint: str) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO ingest_group_runs "
                    "(source_id, group_id, fingerprint, status, error, updated_at) "
                    "VALUES (?, ?, ?, 'done', NULL, ?) "
                    "ON CONFLICT(source_id, group_id) DO UPDATE SET "
                    "fingerprint = excluded.fingerprint, status = 'done', error = NULL, "
                    "updated_at = excluded.updated_at",
                    (source_id, group_id, fingerprint, _now()),
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def mark_group_failed(
        self, *, source_id: str, group_id: str, fingerprint: str, error: str
    ) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO ingest_group_runs "
                    "(source_id, group_id, fingerprint, status, error, updated_at) "
                    "VALUES (?, ?, ?, 'failed', ?, ?) "
                    "ON CONFLICT(source_id, group_id) DO UPDATE SET "
                    "fingerprint = excluded.fingerprint, status = 'failed', "
                    "error = excluded.error, updated_at = excluded.updated_at",
                    (source_id, group_id, fingerprint, error, _now()),
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def remove_source(self, source_id: str) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "DELETE FROM ingest_group_runs WHERE source_id = ?", (source_id,)
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def mark_dead_letter(self, *, path: str, error: str, attempts: int) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO ingest_dead_letters (path, error, attempts, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(path) DO UPDATE SET "
                    "error = excluded.error, attempts = excluded.attempts, "
                    "updated_at = excluded.updated_at",
                    (path, error, attempts, _now()),
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def dead_letters(self) -> list[DeadLetter]:
        rows = self._conn.execute(
            "SELECT path, error, attempts, updated_at FROM ingest_dead_letters "
            "ORDER BY updated_at"
        ).fetchall()
        return [DeadLetter(path=r[0], error=r[1], attempts=r[2], updated_at=r[3]) for r in rows]

    def clear_dead_letter(self, path: str) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM ingest_dead_letters WHERE path = ?", (path,))
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def backup_to(self, path: str | Path) -> None:
        """Online backup via the stdlib sqlite3 backup API — same pattern
        as every other SQLite-backed store in this codebase (see
        ragcore.contextual.sqlite_store.SqliteContextStore.backup_to)."""
        target = sqlite3.connect(str(path))
        try:
            self._conn.backup(target)
        finally:
            target.close()

    def restore_from_backup(self, path: str | Path) -> None:
        with self._lock:
            source = sqlite3.connect(str(path))
            try:
                source.backup(self._conn)
            finally:
                source.close()
