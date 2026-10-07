"""HierarchyStore implementation backed by stdlib sqlite3 — own file, own
connection, copying ragcore.concepts.sqlite_store.SqliteConceptStore's
connection setup and single-transaction-with-checkpoint pattern exactly:
`add_node` writes a node's row and its checkpoint row together, in one
BEGIN IMMEDIATE transaction, so a node is never left checkpointed 'done'
with a missing or stale row.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from ragcore.errors import wrap_backend_errors
from ragcore.hierarchy.types import HierarchyLevel, HierarchyNode

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hierarchy_nodes (
    node_id TEXT PRIMARY KEY,
    level TEXT NOT NULL,
    source_id TEXT NOT NULL,
    label TEXT NOT NULL,
    summary TEXT NOT NULL,
    child_ids_json TEXT NOT NULL,
    source_chunk_ids_json TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    vector_json TEXT,
    tenant_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_hierarchy_nodes_source ON hierarchy_nodes(source_id);
CREATE INDEX IF NOT EXISTS idx_hierarchy_nodes_level ON hierarchy_nodes(level);

CREATE TABLE IF NOT EXISTS hierarchy_runs (
    node_id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_node(row: tuple) -> HierarchyNode:
    (
        node_id, level, source_id, label, summary, child_ids_json,
        source_chunk_ids_json, model, prompt_version, created_at, vector_json,
        tenant_id,
    ) = row
    return HierarchyNode(
        node_id=node_id,
        level=level,
        source_id=source_id,
        label=label,
        summary=summary,
        child_ids=tuple(json.loads(child_ids_json)),
        source_chunk_ids=tuple(json.loads(source_chunk_ids_json)),
        model=model,
        prompt_version=prompt_version,
        created_at=created_at,
        vector=tuple(json.loads(vector_json)) if vector_json else None,
        tenant_id=tenant_id,
    )


_NODE_COLUMNS = (
    "node_id, level, source_id, label, summary, child_ids_json, "
    "source_chunk_ids_json, model, prompt_version, created_at, vector_json, tenant_id"
)


class SqliteHierarchyStore:
    """Default HierarchyStore: one SQLite file, one node table, one
    per-node checkpoint table."""

    @wrap_backend_errors
    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(hierarchy_nodes)")}
        if "tenant_id" not in columns:
            self._conn.execute("ALTER TABLE hierarchy_nodes ADD COLUMN tenant_id TEXT")

    @wrap_backend_errors
    def add_node(self, node: HierarchyNode, *, fingerprint: str) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    f"INSERT INTO hierarchy_nodes ({_NODE_COLUMNS}) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(node_id) DO UPDATE SET "
                    "level = excluded.level, source_id = excluded.source_id, "
                    "label = excluded.label, summary = excluded.summary, "
                    "child_ids_json = excluded.child_ids_json, "
                    "source_chunk_ids_json = excluded.source_chunk_ids_json, "
                    "model = excluded.model, prompt_version = excluded.prompt_version, "
                    "created_at = excluded.created_at, vector_json = excluded.vector_json, "
                    "tenant_id = excluded.tenant_id",
                    (
                        node.node_id, node.level, node.source_id, node.label, node.summary,
                        json.dumps(list(node.child_ids)), json.dumps(list(node.source_chunk_ids)),
                        node.model, node.prompt_version, node.created_at,
                        json.dumps(list(node.vector)) if node.vector is not None else None,
                        node.tenant_id,
                    ),
                )
                self._conn.execute(
                    "INSERT INTO hierarchy_runs (node_id, fingerprint, status, error, updated_at) "
                    "VALUES (?, ?, 'done', NULL, ?) "
                    "ON CONFLICT(node_id) DO UPDATE SET "
                    "fingerprint = excluded.fingerprint, status = 'done', error = NULL, "
                    "updated_at = excluded.updated_at",
                    (node.node_id, fingerprint, _now()),
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def checkpoint(self, node_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT fingerprint FROM hierarchy_runs WHERE node_id = ? AND status = 'done'",
            (node_id,),
        ).fetchone()
        return row[0] if row else None

    @wrap_backend_errors
    def mark_node_failed(self, *, node_id: str, fingerprint: str, error: str) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO hierarchy_runs (node_id, fingerprint, status, error, updated_at) "
                    "VALUES (?, ?, 'failed', ?, ?) "
                    "ON CONFLICT(node_id) DO UPDATE SET "
                    "fingerprint = excluded.fingerprint, status = 'failed', "
                    "error = excluded.error, updated_at = excluded.updated_at",
                    (node_id, fingerprint, error, _now()),
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def node(self, node_id: str) -> HierarchyNode | None:
        row = self._conn.execute(
            f"SELECT {_NODE_COLUMNS} FROM hierarchy_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        return _row_to_node(row) if row else None

    @wrap_backend_errors
    def nodes_for_source(self, source_id: str) -> list[HierarchyNode]:
        rows = self._conn.execute(
            f"SELECT {_NODE_COLUMNS} FROM hierarchy_nodes WHERE source_id = ?", (source_id,)
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    @wrap_backend_errors
    def nodes_by_level(self, level: HierarchyLevel) -> list[HierarchyNode]:
        rows = self._conn.execute(
            f"SELECT {_NODE_COLUMNS} FROM hierarchy_nodes WHERE level = ?", (level,)
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    @wrap_backend_errors
    def all_nodes(self) -> list[HierarchyNode]:
        rows = self._conn.execute(f"SELECT {_NODE_COLUMNS} FROM hierarchy_nodes").fetchall()
        return [_row_to_node(r) for r in rows]

    @wrap_backend_errors
    def remove_source(self, source_id: str) -> None:
        """Deletes every node and checkpoint row belonging to `source_id` —
        used by ragcore.versioning.sync.sync_corpus when a source is removed
        or has changed (and is being re-ingested under a new source_id)."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                node_ids = [
                    r[0]
                    for r in self._conn.execute(
                        "SELECT node_id FROM hierarchy_nodes WHERE source_id = ?", (source_id,)
                    ).fetchall()
                ]
                self._conn.execute("DELETE FROM hierarchy_nodes WHERE source_id = ?", (source_id,))
                if node_ids:
                    placeholders = ",".join("?" * len(node_ids))
                    self._conn.execute(
                        f"DELETE FROM hierarchy_runs WHERE node_id IN ({placeholders})", node_ids
                    )
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
