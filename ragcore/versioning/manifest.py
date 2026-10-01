"""IngestionManifest — a small ledger tracking which file paths have been
ingested, under what content hash and source_id, and when, plus a record of
every snapshot taken for rollback. Own SQLite file, own connection, copying
the same WAL/lock/checkpoint conventions every other store in this codebase
uses.

This is the piece of state Corpus itself deliberately never keeps. A
content-addressed source_id (sha256 of file bytes) means "has this file's
*content* changed?" is already free — but "what source_id did this *path*
resolve to last time?" is not, and that mapping is exactly what
ragcore.versioning.sync.sync_corpus needs to correctly clean up a changed
file's OLD chunks: `Corpus.ingest(replace=True)` deletes by the file's
CURRENT (newly-computed) source_id, which is a no-op the moment a file's
content — and therefore its source_id — has changed, silently orphaning the
old chunks forever. See sync.py's module docstring for the full story.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from ragcore.errors import wrap_backend_errors
from ragcore.versioning.types import Snapshot, SourceState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    path TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    ingested_at TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    tag TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    lancedb_version INTEGER,
    backups_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_file(path: str | Path) -> str:
    """The exact same content-addressing scheme
    ragcore.types.Document.source_id already uses (sha256 of file bytes) —
    reused here, not reimplemented differently, so a manifest-computed hash
    always matches what Corpus.ingest() will compute for the same file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class IngestionManifest:
    @wrap_backend_errors
    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)

    # -- source tracking -------------------------------------------------

    @wrap_backend_errors
    def record_ingest(self, *, path: str, source_id: str, content_hash: str, chunk_count: int) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO sources (path, source_id, content_hash, chunk_count, ingested_at, status) "
                    "VALUES (?, ?, ?, ?, ?, 'active') "
                    "ON CONFLICT(path) DO UPDATE SET "
                    "source_id = excluded.source_id, content_hash = excluded.content_hash, "
                    "chunk_count = excluded.chunk_count, ingested_at = excluded.ingested_at, status = 'active'",
                    (path, source_id, content_hash, chunk_count, _now()),
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def mark_removed(self, path: str) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "UPDATE sources SET status = 'removed', ingested_at = ? WHERE path = ?",
                    (_now(), path),
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def source_id_for_path(self, path: str) -> str | None:
        row = self._conn.execute(
            "SELECT source_id FROM sources WHERE path = ? AND status = 'active'", (path,)
        ).fetchone()
        return row[0] if row else None

    @wrap_backend_errors
    def current_state(self) -> dict[str, SourceState]:
        rows = self._conn.execute(
            "SELECT path, source_id, content_hash, chunk_count, ingested_at "
            "FROM sources WHERE status = 'active'"
        ).fetchall()
        return {
            path: SourceState(
                path=path, source_id=source_id, content_hash=content_hash,
                chunk_count=chunk_count, ingested_at=ingested_at,
            )
            for path, source_id, content_hash, chunk_count, ingested_at in rows
        }

    # -- snapshot tracking -----------------------------------------------

    @wrap_backend_errors
    def begin_snapshot(self, tag: str) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO snapshots (tag, status, lancedb_version, backups_json, created_at) "
                    "VALUES (?, 'pending', NULL, '{}', ?)",
                    (tag, _now()),
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def complete_snapshot(
        self, tag: str, *, lancedb_version: int | None, backups: dict[str, str]
    ) -> Snapshot:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "UPDATE snapshots SET status = 'complete', lancedb_version = ?, backups_json = ? "
                    "WHERE tag = ?",
                    (lancedb_version, json.dumps(backups), tag),
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")
        return self.get_snapshot(tag)

    @wrap_backend_errors
    def get_snapshot(self, tag: str) -> Snapshot | None:
        row = self._conn.execute(
            "SELECT status, lancedb_version, backups_json, created_at FROM snapshots WHERE tag = ?",
            (tag,),
        ).fetchone()
        if row is None:
            return None
        status, lancedb_version, backups_json, created_at = row
        return Snapshot(
            tag=tag, status=status, lancedb_version=lancedb_version,
            backups=json.loads(backups_json), created_at=created_at,
        )

    @wrap_backend_errors
    def list_snapshots(self) -> list[Snapshot]:
        rows = self._conn.execute(
            "SELECT tag, status, lancedb_version, backups_json, created_at FROM snapshots "
            "ORDER BY created_at"
        ).fetchall()
        return [
            Snapshot(tag=tag, status=status, lancedb_version=lancedb_version,
                     backups=json.loads(backups_json), created_at=created_at)
            for tag, status, lancedb_version, backups_json, created_at in rows
        ]

    @wrap_backend_errors
    def backup_to(self, path: str | Path) -> None:
        target = sqlite3.connect(str(path))
        try:
            self._conn.backup(target)
        finally:
            target.close()
