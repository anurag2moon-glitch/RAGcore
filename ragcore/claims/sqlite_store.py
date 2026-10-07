"""ClaimStore implementation backed by stdlib sqlite3 — same connection
setup (WAL, busy_timeout, threading.Lock, BEGIN IMMEDIATE around every
read-then-write merge) every other store in this codebase uses, own file,
own connection.

Unlike ragcore.concepts.sqlite_store.SqliteConceptStore, `source_id` is
stored as a real column on `claims` (not recovered via a chunk_id-prefix
convention) — SourceClaim always carries it, so there's no reason to
reinvent that trick here.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ragcore.claims.types import ClaimRelation, SourceClaim
from ragcore.errors import wrap_backend_errors

_SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    text TEXT NOT NULL,
    author_confidence TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    vector_json TEXT,
    tenant_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_claims_source ON claims(source_id);
CREATE INDEX IF NOT EXISTS idx_claims_chunk ON claims(chunk_id);

CREATE TABLE IF NOT EXISTS claim_concepts (
    claim_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    PRIMARY KEY (claim_id, concept_id)
);
CREATE INDEX IF NOT EXISTS idx_claim_concepts_concept ON claim_concepts(concept_id);

CREATE TABLE IF NOT EXISTS claim_relations (
    relation_id TEXT PRIMARY KEY,
    source_claim_id TEXT NOT NULL,
    target_claim_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    rationale TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claim_relations_source ON claim_relations(source_claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_relations_target ON claim_relations(target_claim_id);

CREATE TABLE IF NOT EXISTS claim_runs (
    source_id TEXT NOT NULL,
    context_group_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (source_id, context_group_id)
);

CREATE TABLE IF NOT EXISTS link_runs (
    concept_id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteClaimStore:
    """Default ClaimStore. See module docstring for the connection/locking
    conventions, shared verbatim with every other SQLite-backed store in
    this codebase."""

    @wrap_backend_errors
    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(claims)")}
        if "tenant_id" not in columns:
            self._conn.execute("ALTER TABLE claims ADD COLUMN tenant_id TEXT")

    # -- extraction writes (single transaction, includes the checkpoint) --

    @wrap_backend_errors
    def add_chapter_claims(
        self,
        *,
        source_id: str,
        context_group_id: str,
        fingerprint: str,
        claims: Sequence[SourceClaim],
    ) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for c in claims:
                    vector_json = json.dumps(list(c.vector)) if c.vector is not None else None
                    self._conn.execute(
                        "INSERT INTO claims "
                        "(claim_id, source_id, chunk_id, text, author_confidence, model, "
                        "prompt_version, created_at, vector_json, tenant_id) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(claim_id) DO UPDATE SET "
                        "vector_json = COALESCE(claims.vector_json, excluded.vector_json)",
                        (
                            c.claim_id, c.source_id, c.chunk_id, c.text, c.author_confidence,
                            c.model, c.prompt_version, c.created_at, vector_json, c.tenant_id,
                        ),
                    )
                    for concept_id in c.concept_ids:
                        self._conn.execute(
                            "INSERT OR IGNORE INTO claim_concepts (claim_id, concept_id) VALUES (?, ?)",
                            (c.claim_id, concept_id),
                        )

                self._conn.execute(
                    "INSERT INTO claim_runs (source_id, context_group_id, fingerprint, status, error, updated_at) "
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
            "SELECT fingerprint FROM claim_runs "
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
                    "INSERT INTO claim_runs (source_id, context_group_id, fingerprint, status, error, updated_at) "
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

    # -- reads: claims ----------------------------------------------------

    @wrap_backend_errors
    def claim(self, claim_id: str) -> SourceClaim | None:
        row = self._conn.execute(
            "SELECT source_id, chunk_id, text, author_confidence, model, prompt_version, "
            "created_at, vector_json, tenant_id FROM claims WHERE claim_id = ?",
            (claim_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_claim(claim_id, row)

    @wrap_backend_errors
    def claims_for_concept(self, concept_id: str) -> list[SourceClaim]:
        rows = self._conn.execute(
            "SELECT c.claim_id, c.source_id, c.chunk_id, c.text, c.author_confidence, c.model, "
            "c.prompt_version, c.created_at, c.vector_json, c.tenant_id "
            "FROM claim_concepts cc JOIN claims c ON cc.claim_id = c.claim_id "
            "WHERE cc.concept_id = ? ORDER BY c.claim_id",
            (concept_id,),
        ).fetchall()
        return [self._row_to_claim(r[0], r[1:]) for r in rows]

    @wrap_backend_errors
    def concept_ids_with_claims(self, *, min_count: int = 2) -> list[str]:
        rows = self._conn.execute(
            "SELECT concept_id FROM claim_concepts GROUP BY concept_id HAVING COUNT(DISTINCT claim_id) >= ?",
            (min_count,),
        ).fetchall()
        return [r[0] for r in rows]

    # -- relations ---------------------------------------------------------

    @wrap_backend_errors
    def put_relations(self, relations: Sequence[ClaimRelation]) -> None:
        if not relations:
            return
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for r in relations:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO claim_relations "
                        "(relation_id, source_claim_id, target_claim_id, relation_type, rationale, "
                        "model, prompt_version, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            r.relation_id, r.source_claim_id, r.target_claim_id, r.relation_type,
                            r.rationale, r.model, r.prompt_version, r.created_at,
                        ),
                    )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def relations_for_claim(self, claim_id: str) -> list[ClaimRelation]:
        rows = self._conn.execute(
            "SELECT relation_id, source_claim_id, target_claim_id, relation_type, rationale, "
            "model, prompt_version, created_at FROM claim_relations "
            "WHERE source_claim_id = ? OR target_claim_id = ?",
            (claim_id, claim_id),
        ).fetchall()
        return [
            ClaimRelation(
                relation_id=relation_id, source_claim_id=source_claim_id,
                target_claim_id=target_claim_id, relation_type=relation_type,
                rationale=rationale, model=model, prompt_version=prompt_version, created_at=created_at,
            )
            for relation_id, source_claim_id, target_claim_id, relation_type, rationale,
            model, prompt_version, created_at in rows
        ]

    @wrap_backend_errors
    def link_checkpoint(self, concept_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT fingerprint FROM link_runs WHERE concept_id = ? AND status = 'done'",
            (concept_id,),
        ).fetchone()
        return row[0] if row else None

    @wrap_backend_errors
    def mark_concept_linked(self, *, concept_id: str, fingerprint: str) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO link_runs (concept_id, fingerprint, status, updated_at) "
                    "VALUES (?, ?, 'done', ?) "
                    "ON CONFLICT(concept_id) DO UPDATE SET "
                    "fingerprint = excluded.fingerprint, status = 'done', updated_at = excluded.updated_at",
                    (concept_id, fingerprint, _now()),
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    # -- removal (incremental ingestion) -----------------------------------

    @wrap_backend_errors
    def remove_chunk_ids(self, chunk_ids: Sequence[str]) -> None:
        """Deletes every claim from the given chunk_ids, cascading to
        claim_concepts and to every claim_relations row touching a deleted
        claim in EITHER direction — the same rule
        ragcore.concepts.sqlite_store.SqliteConceptStore.remove_chunk_ids
        applies when an edge's endpoint concept vanishes, even though the
        relation's *other* endpoint (and its own source) may be untouched.
        Used by ragcore.versioning.sync."""
        if not chunk_ids:
            return
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                placeholders = ",".join("?" * len(chunk_ids))
                ids = list(chunk_ids)
                self._conn.execute(
                    f"DELETE FROM claim_concepts WHERE claim_id IN "
                    f"(SELECT claim_id FROM claims WHERE chunk_id IN ({placeholders}))",
                    ids,
                )
                self._conn.execute(
                    f"DELETE FROM claim_relations WHERE source_claim_id IN "
                    f"(SELECT claim_id FROM claims WHERE chunk_id IN ({placeholders})) "
                    f"OR target_claim_id IN "
                    f"(SELECT claim_id FROM claims WHERE chunk_id IN ({placeholders}))",
                    ids + ids,
                )
                self._conn.execute(f"DELETE FROM claims WHERE chunk_id IN ({placeholders})", ids)

                # A concept now below the 2-claim minimum can no longer be
                # linked meaningfully — garbage-collect its stale link
                # checkpoint so a future link_claims() run doesn't skip it
                # forever if claims are later re-added under the same id.
                self._conn.execute(
                    "DELETE FROM link_runs WHERE concept_id NOT IN "
                    "(SELECT concept_id FROM claim_concepts GROUP BY concept_id HAVING COUNT(DISTINCT claim_id) >= 2)"
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def remove_source(self, source_id: str) -> None:
        """All claims belonging to `source_id`, pruned via
        remove_chunk_ids, plus that source's own extraction checkpoints
        from claim_runs (otherwise a re-ingest under the same source_id,
        rare but possible, would wrongly be skipped as already-done)."""
        chunk_ids = [
            r[0]
            for r in self._conn.execute(
                "SELECT DISTINCT chunk_id FROM claims WHERE source_id = ?", (source_id,)
            ).fetchall()
        ]
        self.remove_chunk_ids(chunk_ids)

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM claim_runs WHERE source_id = ?", (source_id,))
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
        in place from a previously-taken backup file. Used by
        ragcore.versioning.snapshot.rollback."""
        with self._lock:
            source = sqlite3.connect(str(path))
            try:
                source.backup(self._conn)
            finally:
                source.close()

    # -- internals -----------------------------------------------------

    def _row_to_claim(self, claim_id: str, row: tuple) -> SourceClaim:
        (
            source_id, chunk_id, text, author_confidence, model, prompt_version, created_at,
            vector_json, tenant_id,
        ) = row
        concept_ids = [
            r[0]
            for r in self._conn.execute(
                "SELECT concept_id FROM claim_concepts WHERE claim_id = ?", (claim_id,)
            ).fetchall()
        ]
        return SourceClaim(
            claim_id=claim_id,
            concept_ids=tuple(concept_ids),
            source_id=source_id,
            chunk_id=chunk_id,
            text=text,
            author_confidence=author_confidence,
            model=model,
            prompt_version=prompt_version,
            created_at=created_at,
            vector=tuple(json.loads(vector_json)) if vector_json else None,
            tenant_id=tenant_id,
        )
