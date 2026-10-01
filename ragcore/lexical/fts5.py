"""LexicalStore implementation backed by SQLite FTS5 — embedded, zero-ops,
zero new dependency (FTS5 ships compiled into every standard CPython sqlite3
build; confirmed present and its bm25() ranking function queryable in this
project's environment).

Deliberately a single plain FTS5 table, not FTS5 "external content" mode:
external-content's main benefits (supporting the `'rebuild'` command;
avoiding a second copy of text already owned by another table) don't apply
here — ragcore is the only writer to this file, and every write goes
through `index()`/`delete()` directly, so there's never a need to rebuild
from an out-of-band content table.

FTS5's bm25() score is a small negative number (lower is better) and is
never surfaced past this module — ragcore.retrieval.hybrid.HybridRetriever
consumes only LexicalHit.rank (1-based, best first) for reciprocal rank
fusion, never a raw score, so FTS5's sign/scale convention is a non-issue.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Sequence

from ragcore.errors import StoreError, wrap_backend_errors
from ragcore.lexical.types import LexicalHit
from ragcore.types import Chunk

_WORD_RE = re.compile(r"[a-z0-9]+")


def _fts5_query(text: str) -> str | None:
    """Tokenizes `text` to bare alphanumeric words and joins them with OR,
    each individually double-quoted. This is a deliberate sanitization, not
    just a convenience: FTS5's query syntax (AND/OR/NOT/NEAR, column
    filters, prefix `*`, unbalanced quotes) would otherwise let a raw user
    query raise sqlite3.OperationalError or silently mean something
    different than "search for these words". OR (not AND) maximizes recall
    on the lexical channel, which exists specifically to catch keyword
    matches the vector channel might miss — HybridRetriever's fusion, not
    this store, is responsible for weighting/ranking across channels.
    Returns None if the query has no searchable tokens at all.
    """
    tokens = _WORD_RE.findall(text.lower())
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


class Fts5LexicalStore:
    """Default LexicalStore: one SQLite file, one FTS5 virtual table.

    Concurrency: same rationale as ragcore.graph.sqlite_store.SqliteGraphStore
    — `check_same_thread=False` plus a `threading.Lock` serializes concurrent
    callers sharing one instance; `isolation_level=None` (autocommit) with an
    explicit `BEGIN IMMEDIATE` around each write acquires SQLite's write lock
    before the read/write, so concurrent separate connections to the same
    file still serialize correctly; `busy_timeout` makes a writer wait for
    that lock instead of raising immediately.
    """

    @wrap_backend_errors
    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5("
            "chunk_id UNINDEXED, source_id UNINDEXED, text, "
            "tokenize='porter unicode61')"
        )

    @wrap_backend_errors
    def index(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                ids = [c.chunk_id for c in chunks]
                placeholders = ",".join("?" * len(ids))
                self._conn.execute(
                    f"DELETE FROM fts_chunks WHERE chunk_id IN ({placeholders})", ids
                )
                self._conn.executemany(
                    "INSERT INTO fts_chunks (chunk_id, source_id, text) VALUES (?, ?, ?)",
                    [(c.chunk_id, c.meta.source_id, c.text) for c in chunks],
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def search(
        self, query: str, k: int, *, source_ids: Sequence[str] | None = None
    ) -> list[LexicalHit]:
        fts_query = _fts5_query(query)
        if fts_query is None or k <= 0:
            return []

        sql = "SELECT chunk_id FROM fts_chunks WHERE fts_chunks MATCH ?"
        params: list[object] = [fts_query]
        if source_ids:
            placeholders = ",".join("?" * len(source_ids))
            sql += f" AND source_id IN ({placeholders})"
            params.extend(source_ids)
        sql += " ORDER BY rank LIMIT ?"
        params.append(k)

        rows = self._conn.execute(sql, params).fetchall()
        return [LexicalHit(chunk_id=row[0], rank=i) for i, row in enumerate(rows, start=1)]

    @wrap_backend_errors
    def delete(
        self, *, source_id: str | None = None, chunk_ids: Sequence[str] | None = None
    ) -> int:
        if (source_id is None) == (not chunk_ids):
            raise StoreError("delete() requires exactly one of source_id or chunk_ids")

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                before = self._count_locked()
                if source_id is not None:
                    self._conn.execute("DELETE FROM fts_chunks WHERE source_id = ?", (source_id,))
                else:
                    placeholders = ",".join("?" * len(chunk_ids))
                    self._conn.execute(
                        f"DELETE FROM fts_chunks WHERE chunk_id IN ({placeholders})",
                        list(chunk_ids),
                    )
                after = self._count_locked()
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")
        return before - after

    @wrap_backend_errors
    def count(self) -> int:
        return self._count_locked()

    def _count_locked(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM fts_chunks").fetchone()[0]

    @wrap_backend_errors
    def backup_to(self, path: str | Path) -> None:
        """Online backup via the stdlib sqlite3 backup API — safe under
        concurrent readers in WAL mode, unlike a plain file copy (which can
        capture a torn snapshot of a live WAL-mode database). Used by
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
