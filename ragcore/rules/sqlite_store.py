"""RuleStore implementation backed by stdlib sqlite3 — same connection/
locking conventions as every other SQLite-backed store in this
codebase (WAL mode, `threading.Lock`, `BEGIN IMMEDIATE` around every
write). See `ragcore.rules.base.RuleStore`'s own docstring for the
immutable-version-on-edit write discipline this implements.

A `Rule`'s full content is persisted as one `payload_json` blob
(`Rule.model_dump_json()`/`Rule.model_validate_json()`) rather than
exploded into one column per field — unlike `ChunkMeta`'s denormalized
columns in `LanceDBStore`, a `Rule`'s nested `conditions`/`effect`/
`remedy` trees have no fixed shape the Filter DSL needs to index into;
only the handful of scalar fields this store's own query methods
actually filter by (`book_id`, `domain`, `is_latest`, ...) are
denormalized into real columns.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ragcore.errors import RagcoreError, wrap_backend_errors
from ragcore.rules.types import Rule

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rules (
    rule_id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    supersedes TEXT,
    is_latest INTEGER NOT NULL DEFAULT 1,
    domain TEXT NOT NULL,
    book_id TEXT NOT NULL,
    review_status TEXT NOT NULL,
    tenant_scope TEXT NOT NULL,
    tenant_id TEXT,
    confidence REAL NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rules_book ON rules(book_id);
CREATE INDEX IF NOT EXISTS idx_rules_domain ON rules(domain);
CREATE INDEX IF NOT EXISTS idx_rules_latest ON rules(is_latest);

CREATE TABLE IF NOT EXISTS rule_runs (
    source_id TEXT NOT NULL,
    context_group_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (source_id, context_group_id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteRuleStore:
    """Default RuleStore: one SQLite file, one rules table (payload as
    JSON + denormalized scalar filter columns), one per-chapter
    checkpoint table."""

    @wrap_backend_errors
    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)

    @wrap_backend_errors
    def add_rules(self, rules: Sequence[Rule]) -> None:
        if not rules:
            return
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for rule in rules:
                    self._insert_locked(rule)
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def _insert_locked(self, rule: Rule) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO rules "
            "(rule_id, version, supersedes, is_latest, domain, book_id, "
            "review_status, tenant_scope, tenant_id, confidence, payload_json) "
            "VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)",
            (
                rule.rule_id, rule.version, rule.supersedes, rule.domain, rule.book_id,
                rule.review_status, rule.tenant_scope, rule.tenant_id, rule.confidence,
                rule.model_dump_json(),
            ),
        )

    @wrap_backend_errors
    def get(self, rule_id: str) -> Rule | None:
        row = self._conn.execute(
            "SELECT payload_json FROM rules WHERE rule_id = ?", (rule_id,)
        ).fetchone()
        return Rule.model_validate_json(row[0]) if row else None

    @wrap_backend_errors
    def rules_for_book(self, book_id: str, *, latest_only: bool = True) -> list[Rule]:
        sql = "SELECT payload_json FROM rules WHERE book_id = ?"
        params: list[object] = [book_id]
        if latest_only:
            sql += " AND is_latest = 1"
        rows = self._conn.execute(sql, params).fetchall()
        return [Rule.model_validate_json(r[0]) for r in rows]

    @wrap_backend_errors
    def rules_for_domain(self, domain: str, *, latest_only: bool = True) -> list[Rule]:
        sql = "SELECT payload_json FROM rules WHERE domain = ?"
        params: list[object] = [domain]
        if latest_only:
            sql += " AND is_latest = 1"
        rows = self._conn.execute(sql, params).fetchall()
        return [Rule.model_validate_json(r[0]) for r in rows]

    @wrap_backend_errors
    def supersede(self, old_rule_id: str, new_rule: Rule) -> None:
        if new_rule.supersedes != old_rule_id:
            raise RagcoreError(
                f"new_rule.supersedes ({new_rule.supersedes!r}) must equal "
                f"old_rule_id ({old_rule_id!r})"
            )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                old_row = self._conn.execute(
                    "SELECT rule_id FROM rules WHERE rule_id = ?", (old_rule_id,)
                ).fetchone()
                if old_row is None:
                    raise RagcoreError(f"no rule found with rule_id {old_rule_id!r}")
                self._insert_locked(new_rule)
                self._conn.execute(
                    "UPDATE rules SET is_latest = 0 WHERE rule_id = ?", (old_rule_id,)
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    # -- checkpointing (per-chapter, same shape as every other extraction layer) --

    @wrap_backend_errors
    def checkpoint(self, source_id: str, context_group_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT fingerprint FROM rule_runs "
            "WHERE source_id = ? AND context_group_id = ? AND status = 'done'",
            (source_id, context_group_id),
        ).fetchone()
        return row[0] if row else None

    @wrap_backend_errors
    def mark_chapter_done(
        self, *, source_id: str, context_group_id: str, fingerprint: str
    ) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO rule_runs "
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
    def mark_chapter_failed(
        self, *, source_id: str, context_group_id: str, fingerprint: str, error: str
    ) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO rule_runs "
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
    def backup_to(self, path: str | Path) -> None:
        """Online backup via the stdlib sqlite3 backup API — same
        reasoning as every other SQLite-backed store in this codebase
        (safe under concurrent readers in WAL mode, unlike a plain
        file copy)."""
        target = sqlite3.connect(str(path))
        try:
            self._conn.backup(target)
        finally:
            target.close()

    @wrap_backend_errors
    def restore_from_backup(self, path: str | Path) -> None:
        with self._lock:
            source = sqlite3.connect(str(path))
            try:
                source.backup(self._conn)
            finally:
                source.close()
