"""ContextStore implementation backed by stdlib sqlite3 — own file, own
connection, copying ragcore.concepts.sqlite_store.SqliteConceptStore's
connection setup and single-transaction-with-checkpoint pattern exactly.

The checkpoint shape is simpler than concepts' — there is no merge-vs-new
distinction here (a blurb is either written or it isn't; content-addressing
by chunk_id already makes a re-write of the same chunk idempotent), so
`add_chapter_blurbs` is a plain upsert-per-chunk plus the checkpoint row, in
one BEGIN IMMEDIATE transaction — same reasoning as
ragcore.concepts.sqlite_store's module docstring: a chapter must never be
left checkpointed 'done' with some of its blurbs missing.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from ragcore.errors import wrap_backend_errors


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS context_blurbs (
    chunk_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    blurb TEXT NOT NULL,
    tenant_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_context_blurbs_source ON context_blurbs(source_id);

CREATE TABLE IF NOT EXISTS context_runs (
    source_id TEXT NOT NULL,
    context_group_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (source_id, context_group_id)
);
"""


class SqliteContextStore:
    """Default ContextStore: one SQLite file, blurbs keyed by chunk_id, a
    per-chapter checkpoint mirroring ragcore.concepts.sqlite_store's
    concept_runs table."""

    @wrap_backend_errors
    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(context_blurbs)")}
        if "tenant_id" not in columns:
            self._conn.execute("ALTER TABLE context_blurbs ADD COLUMN tenant_id TEXT")

    @wrap_backend_errors
    def add_chapter_blurbs(
        self,
        *,
        source_id: str,
        context_group_id: str,
        fingerprint: str,
        blurbs: dict[str, str],
        tenant_id: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.executemany(
                    "INSERT INTO context_blurbs (chunk_id, source_id, blurb, tenant_id) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(chunk_id) DO UPDATE SET blurb = excluded.blurb, "
                    "tenant_id = excluded.tenant_id",
                    [(chunk_id, source_id, blurb, tenant_id) for chunk_id, blurb in blurbs.items()],
                )
                self._conn.execute(
                    "INSERT INTO context_runs "
                    "(source_id, context_group_id, fingerprint, status, error, updated_at) "
                    "VALUES (?, ?, ?, 'done', NULL, ?) "
                    "ON CONFLICT(source_id, context_group_id) DO UPDATE SET "
                    "fingerprint = excluded.fingerprint, status = 'done', error = NULL, "
                    "updated_at = excluded.updated_at",
                    (source_id, context_group_id, fingerprint, _now()),
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def checkpoint(self, source_id: str, context_group_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT fingerprint FROM context_runs "
            "WHERE source_id = ? AND context_group_id = ? AND status = 'done'",
            (source_id, context_group_id),
        ).fetchone()
        return row[0] if row else None

    @wrap_backend_errors
    def mark_chapter_failed(
        self, *, source_id: str, context_group_id: str, fingerprint: str, error: str
    ) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO context_runs "
                    "(source_id, context_group_id, fingerprint, status, error, updated_at) "
                    "VALUES (?, ?, ?, 'failed', ?, ?) "
                    "ON CONFLICT(source_id, context_group_id) DO UPDATE SET "
                    "fingerprint = excluded.fingerprint, status = 'failed', "
                    "error = excluded.error, updated_at = excluded.updated_at",
                    (source_id, context_group_id, fingerprint, error, _now()),
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def blurb(self, chunk_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT blurb FROM context_blurbs WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return row[0] if row else None

    @wrap_backend_errors
    def remove_source(self, source_id: str) -> None:
        """Deletes every blurb and checkpoint row belonging to `source_id` —
        used by ragcore.versioning.sync.sync_corpus when a source is removed
        or has changed (and is being re-ingested under a new source_id)."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM context_blurbs WHERE source_id = ?", (source_id,))
                self._conn.execute("DELETE FROM context_runs WHERE source_id = ?", (source_id,))
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def backup_to(self, path: str | Path) -> None:
        """Online backup via the stdlib sqlite3 backup API — safe under
        concurrent readers in WAL mode, unlike a plain file copy. Used by
        ragcore.versioning.snapshot for basic ingestion rollback."""
        target = sqlite3.connect(str(path))
        try:
            self._conn.backup(target)
        finally:
            target.close()

    @wrap_backend_errors
    def restore_from_backup(self, path: str | Path) -> None:
        """The reverse of backup_to: restores this store's live connection
        in place from a previously-taken backup file — see
        ragcore.graph.sqlite_store.SqliteGraphStore.restore_from_backup for
        the full rationale (identical here). Used by
        ragcore.versioning.snapshot.rollback."""
        with self._lock:
            source = sqlite3.connect(str(path))
            try:
                source.backup(self._conn)
            finally:
                source.close()
