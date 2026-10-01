"""ConceptStore implementation backed by stdlib sqlite3 — same file
SqliteGraphStore uses, own tables, own connection.

Copies ragcore.graph.sqlite_store.SqliteGraphStore's connection setup
verbatim (WAL, busy_timeout, a threading.Lock, BEGIN IMMEDIATE around every
read-then-write merge) but never shares its connection object — writes
through a different connection would sit outside that store's lock. See
ragcore.concepts.pipeline.build_concepts for why add_chapter_concepts must
be one transaction including the checkpoint row, and why the extraction
LLM call must never happen while that transaction is open.

Edge weight is deliberately never stored as an incrementally-updated
counter — that would double-count if the same chapter is ever
re-processed (e.g. after a fingerprint bump), breaking the idempotency
every other write in this file relies on. Instead concept_edge_chunks (a
chunk_id per edge, INSERT OR IGNORE — a plain idempotent set union, exactly
like concept_chunks) is the source of truth, and weight is derived as
COUNT(DISTINCT chunk_id) whenever an edge is read.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ragcore.concepts.types import Concept, ConceptEdge, CommunitySummary
from ragcore.errors import wrap_backend_errors

_SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts (
    concept_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    description TEXT NOT NULL,
    vector_json TEXT,
    extra_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS concept_chunks (
    concept_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (concept_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_concept_chunks_chunk ON concept_chunks(chunk_id);

CREATE TABLE IF NOT EXISTS concept_edges (
    edge_id TEXT PRIMARY KEY,
    source_concept_id TEXT NOT NULL,
    target_concept_id TEXT NOT NULL,
    extra_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_concept_edges_source ON concept_edges(source_concept_id);
CREATE INDEX IF NOT EXISTS idx_concept_edges_target ON concept_edges(target_concept_id);

CREATE TABLE IF NOT EXISTS concept_edge_chunks (
    edge_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (edge_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS concept_runs (
    source_id TEXT NOT NULL,
    context_group_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    concept_count INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (source_id, context_group_id)
);

CREATE TABLE IF NOT EXISTS concept_communities (
    concept_id TEXT PRIMARY KEY,
    community_id TEXT NOT NULL,
    run_fingerprint TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_concept_communities_community ON concept_communities(community_id);

CREATE TABLE IF NOT EXISTS community_runs (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    fingerprint TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS community_summaries (
    community_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    summary TEXT NOT NULL,
    agreements_json TEXT NOT NULL,
    divergences_json TEXT NOT NULL,
    concept_ids_json TEXT NOT NULL,
    source_chunk_ids_json TEXT NOT NULL,
    source_ids_json TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    vector_json TEXT,
    chunk_fingerprint TEXT
);
"""


def _merge_extra(old: dict, new: dict) -> dict:
    """Plain overwrite-merge: concepts/edges carry no accumulating list keys
    today (unlike ragcore.graph.sqlite_store's aliases/relations). If a
    future extra key needs set-union-across-merges semantics, give this the
    same _LIST_MERGE_KEYS treatment as the graph store."""
    merged = dict(old)
    merged.update(new)
    return merged


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_summary(community_id: str, row: tuple) -> CommunitySummary:
    (
        label, summary, agreements_json, divergences_json, concept_ids_json,
        source_chunk_ids_json, source_ids_json, model, prompt_version, created_at, vector_json,
        chunk_fingerprint,
    ) = row
    return CommunitySummary(
        community_id=community_id,
        label=label,
        summary=summary,
        chunk_fingerprint=chunk_fingerprint,
        agreements=tuple(json.loads(agreements_json)),
        divergences=tuple(json.loads(divergences_json)),
        concept_ids=tuple(json.loads(concept_ids_json)),
        source_chunk_ids=tuple(json.loads(source_chunk_ids_json)),
        source_ids=tuple(json.loads(source_ids_json)),
        model=model,
        prompt_version=prompt_version,
        created_at=created_at,
        vector=tuple(json.loads(vector_json)) if vector_json else None,
    )


class SqliteConceptStore:
    """Default ConceptStore: same SQLite file a SqliteGraphStore may point
    at, its own tables, its own connection.

    Concurrency: identical rationale to SqliteGraphStore —
    `check_same_thread=False` plus a `threading.Lock` serializes concurrent
    callers sharing one instance; `isolation_level=None` (autocommit) with
    an explicit `BEGIN IMMEDIATE` around each merge acquires SQLite's write
    lock before the read, so concurrent separate connections to the same
    file still serialize correctly; `busy_timeout` makes a writer wait for
    that lock instead of raising immediately.
    """

    @wrap_backend_errors
    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)

    # -- extraction writes (single transaction, includes the checkpoint) --

    @wrap_backend_errors
    def add_chapter_concepts(
        self,
        *,
        source_id: str,
        context_group_id: str,
        fingerprint: str,
        concepts: Sequence[Concept],
        edges: Sequence[ConceptEdge],
    ) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for c in concepts:
                    row = self._conn.execute(
                        "SELECT vector_json, extra_json FROM concepts WHERE concept_id = ?",
                        (c.concept_id,),
                    ).fetchone()
                    new_vector_json = json.dumps(list(c.vector)) if c.vector is not None else None
                    if row is None:
                        self._conn.execute(
                            "INSERT INTO concepts (concept_id, label, description, vector_json, extra_json) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (c.concept_id, c.label, c.description, new_vector_json, json.dumps(c.extra)),
                        )
                    else:
                        existing_vector_json, extra_json = row
                        merged_extra = _merge_extra(json.loads(extra_json), c.extra)
                        # Never clobber a stored vector with None — a merge
                        # candidate carries no vector of its own (see
                        # Concept.vector's docstring).
                        vector_json = new_vector_json or existing_vector_json
                        self._conn.execute(
                            "UPDATE concepts SET label = ?, description = ?, vector_json = ?, extra_json = ? "
                            "WHERE concept_id = ?",
                            (c.label, c.description, vector_json, json.dumps(merged_extra), c.concept_id),
                        )
                    for chunk_id in c.source_chunk_ids:
                        self._conn.execute(
                            "INSERT OR IGNORE INTO concept_chunks (concept_id, chunk_id) VALUES (?, ?)",
                            (c.concept_id, chunk_id),
                        )

                for e in edges:
                    row = self._conn.execute(
                        "SELECT extra_json FROM concept_edges WHERE edge_id = ?",
                        (e.edge_id,),
                    ).fetchone()
                    if row is None:
                        self._conn.execute(
                            "INSERT INTO concept_edges "
                            "(edge_id, source_concept_id, target_concept_id, extra_json) "
                            "VALUES (?, ?, ?, ?)",
                            (e.edge_id, e.source_concept_id, e.target_concept_id, json.dumps(e.extra)),
                        )
                    else:
                        (extra_json,) = row
                        merged_extra = _merge_extra(json.loads(extra_json), e.extra)
                        self._conn.execute(
                            "UPDATE concept_edges SET extra_json = ? WHERE edge_id = ?",
                            (json.dumps(merged_extra), e.edge_id),
                        )
                    for chunk_id in e.source_chunk_ids:
                        self._conn.execute(
                            "INSERT OR IGNORE INTO concept_edge_chunks (edge_id, chunk_id) VALUES (?, ?)",
                            (e.edge_id, chunk_id),
                        )

                self._conn.execute(
                    "INSERT INTO concept_runs "
                    "(source_id, context_group_id, fingerprint, status, concept_count, error, updated_at) "
                    "VALUES (?, ?, ?, 'done', ?, NULL, ?) "
                    "ON CONFLICT(source_id, context_group_id) DO UPDATE SET "
                    "fingerprint = excluded.fingerprint, status = 'done', "
                    "concept_count = excluded.concept_count, error = NULL, updated_at = excluded.updated_at",
                    (source_id, context_group_id, fingerprint, len(concepts), _now()),
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def checkpoint(self, source_id: str, context_group_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT fingerprint FROM concept_runs "
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
                    "INSERT INTO concept_runs "
                    "(source_id, context_group_id, fingerprint, status, concept_count, error, updated_at) "
                    "VALUES (?, ?, ?, 'failed', 0, ?, ?) "
                    "ON CONFLICT(source_id, context_group_id) DO UPDATE SET "
                    "fingerprint = excluded.fingerprint, status = 'failed', concept_count = 0, "
                    "error = excluded.error, updated_at = excluded.updated_at",
                    (source_id, context_group_id, fingerprint, error, _now()),
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    # -- reads: concepts ------------------------------------------------

    @wrap_backend_errors
    def concept(self, concept_id: str) -> Concept | None:
        row = self._conn.execute(
            "SELECT label, description, vector_json, extra_json FROM concepts WHERE concept_id = ?",
            (concept_id,),
        ).fetchone()
        if row is None:
            return None
        label, description, vector_json, extra_json = row
        chunk_ids = self._chunk_ids_for_concepts([concept_id]).get(concept_id, [])
        return Concept(
            concept_id=concept_id,
            label=label,
            description=description,
            source_chunk_ids=tuple(chunk_ids),
            extra=json.loads(extra_json) if extra_json else {},
            vector=tuple(json.loads(vector_json)) if vector_json else None,
        )

    @wrap_backend_errors
    def all_concepts(self) -> list[Concept]:
        rows = self._conn.execute("SELECT concept_id FROM concepts").fetchall()
        return [c for (concept_id,) in rows if (c := self.concept(concept_id)) is not None]

    @wrap_backend_errors
    def concepts_for_chunks(self, chunk_ids: Sequence[str]) -> dict[str, list[Concept]]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT cc.chunk_id, c.concept_id, c.label, c.description, c.vector_json, c.extra_json "
            f"FROM concept_chunks cc JOIN concepts c ON cc.concept_id = c.concept_id "
            f"WHERE cc.chunk_id IN ({placeholders})",
            list(chunk_ids),
        ).fetchall()
        concept_chunk_ids = self._chunk_ids_for_concepts({r[1] for r in rows})
        out: dict[str, list[Concept]] = {}
        for chunk_id, concept_id, label, description, vector_json, extra_json in rows:
            out.setdefault(chunk_id, []).append(
                Concept(
                    concept_id=concept_id,
                    label=label,
                    description=description,
                    source_chunk_ids=tuple(concept_chunk_ids.get(concept_id, [])),
                    extra=json.loads(extra_json) if extra_json else {},
                    vector=tuple(json.loads(vector_json)) if vector_json else None,
                )
            )
        return out

    @wrap_backend_errors
    def chunks_for_concepts(self, concept_ids: Sequence[str]) -> dict[str, list[str]]:
        return self._chunk_ids_for_concepts(concept_ids)

    @wrap_backend_errors
    def concept_vector(self, concept_id: str) -> tuple[float, ...] | None:
        row = self._conn.execute(
            "SELECT vector_json FROM concepts WHERE concept_id = ?", (concept_id,)
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return tuple(json.loads(row[0]))

    # -- reads: edges ----------------------------------------------------

    @wrap_backend_errors
    def all_edges(self) -> list[ConceptEdge]:
        rows = self._conn.execute(
            "SELECT edge_id, source_concept_id, target_concept_id, extra_json FROM concept_edges"
        ).fetchall()
        weights = dict(
            self._conn.execute(
                "SELECT edge_id, COUNT(DISTINCT chunk_id) FROM concept_edge_chunks GROUP BY edge_id"
            ).fetchall()
        )
        chunk_ids_by_edge = self._chunk_ids_for_edges([r[0] for r in rows])
        return [
            ConceptEdge(
                edge_id=edge_id,
                source_concept_id=source,
                target_concept_id=target,
                weight=weights.get(edge_id, 0),
                source_chunk_ids=tuple(chunk_ids_by_edge.get(edge_id, [])),
                extra=json.loads(extra_json) if extra_json else {},
            )
            for edge_id, source, target, extra_json in rows
        ]

    # -- communities -------------------------------------------------------

    @wrap_backend_errors
    def community_run_fingerprint(self) -> str | None:
        row = self._conn.execute("SELECT fingerprint FROM community_runs WHERE id = 1").fetchone()
        return row[0] if row else None

    @wrap_backend_errors
    def set_communities(self, *, run_fingerprint: str, assignments: dict[str, str]) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM concept_communities")
                self._conn.executemany(
                    "INSERT INTO concept_communities (concept_id, community_id, run_fingerprint) "
                    "VALUES (?, ?, ?)",
                    [(cid, community_id, run_fingerprint) for cid, community_id in assignments.items()],
                )
                self._conn.execute(
                    "INSERT INTO community_runs (id, fingerprint, updated_at) VALUES (1, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET fingerprint = excluded.fingerprint, "
                    "updated_at = excluded.updated_at",
                    (run_fingerprint, _now()),
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def community_id_for_concept(self, concept_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT community_id FROM concept_communities WHERE concept_id = ?", (concept_id,)
        ).fetchone()
        return row[0] if row else None

    @wrap_backend_errors
    def concept_ids_for_community(self, community_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT concept_id FROM concept_communities WHERE community_id = ?", (community_id,)
        ).fetchall()
        return [r[0] for r in rows]

    @wrap_backend_errors
    def all_community_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT DISTINCT community_id FROM concept_communities").fetchall()
        return [r[0] for r in rows]

    @wrap_backend_errors
    def community_summary(self, community_id: str) -> CommunitySummary | None:
        row = self._conn.execute(
            "SELECT label, summary, agreements_json, divergences_json, concept_ids_json, "
            "source_chunk_ids_json, source_ids_json, model, prompt_version, created_at, vector_json, "
            "chunk_fingerprint "
            "FROM community_summaries WHERE community_id = ?",
            (community_id,),
        ).fetchone()
        return None if row is None else _row_to_summary(community_id, row)

    @wrap_backend_errors
    def put_community_summary(self, summary: CommunitySummary) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO community_summaries "
                    "(community_id, label, summary, agreements_json, divergences_json, concept_ids_json, "
                    "source_chunk_ids_json, source_ids_json, model, prompt_version, created_at, vector_json, "
                    "chunk_fingerprint) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(community_id) DO UPDATE SET "
                    "label = excluded.label, summary = excluded.summary, "
                    "agreements_json = excluded.agreements_json, divergences_json = excluded.divergences_json, "
                    "concept_ids_json = excluded.concept_ids_json, "
                    "source_chunk_ids_json = excluded.source_chunk_ids_json, "
                    "source_ids_json = excluded.source_ids_json, model = excluded.model, "
                    "prompt_version = excluded.prompt_version, created_at = excluded.created_at, "
                    "vector_json = excluded.vector_json, chunk_fingerprint = excluded.chunk_fingerprint",
                    (
                        summary.community_id, summary.label, summary.summary,
                        json.dumps(list(summary.agreements)), json.dumps(list(summary.divergences)),
                        json.dumps(list(summary.concept_ids)), json.dumps(list(summary.source_chunk_ids)),
                        json.dumps(list(summary.source_ids)), summary.model, summary.prompt_version,
                        summary.created_at,
                        json.dumps(list(summary.vector)) if summary.vector is not None else None,
                        summary.chunk_fingerprint,
                    ),
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def communities_for_concepts(self, concept_ids: Sequence[str]) -> list[CommunitySummary]:
        if not concept_ids:
            return []
        placeholders = ",".join("?" * len(concept_ids))
        rows = self._conn.execute(
            f"SELECT DISTINCT community_id FROM concept_communities WHERE concept_id IN ({placeholders})",
            list(concept_ids),
        ).fetchall()
        out: list[CommunitySummary] = []
        for (community_id,) in rows:
            summary = self.community_summary(community_id)
            if summary is not None:
                out.append(summary)
        return out

    # -- removal (item 7: incremental ingestion) -----------------------

    @wrap_backend_errors
    def remove_chunk_ids(self, chunk_ids: Sequence[str]) -> None:
        """Prunes the given chunk_ids from every concept's/edge's
        provenance, deleting any concept or edge left with zero remaining
        chunk associations — and, when a concept is deleted, its
        `concept_communities` assignment along with it (a stale community
        member would otherwise linger until the next detect_communities()
        run notices the concept no longer exists). Used by
        ragcore.versioning.sync.sync_corpus."""
        if not chunk_ids:
            return
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                placeholders = ",".join("?" * len(chunk_ids))
                ids = list(chunk_ids)
                self._conn.execute(
                    f"DELETE FROM concept_chunks WHERE chunk_id IN ({placeholders})", ids
                )
                self._conn.execute(
                    f"DELETE FROM concept_edge_chunks WHERE chunk_id IN ({placeholders})", ids
                )

                orphan_concepts = [
                    r[0]
                    for r in self._conn.execute(
                        "SELECT concept_id FROM concepts WHERE concept_id NOT IN "
                        "(SELECT DISTINCT concept_id FROM concept_chunks)"
                    ).fetchall()
                ]
                if orphan_concepts:
                    op = ",".join("?" * len(orphan_concepts))
                    self._conn.execute(f"DELETE FROM concepts WHERE concept_id IN ({op})", orphan_concepts)
                    self._conn.execute(
                        f"DELETE FROM concept_communities WHERE concept_id IN ({op})", orphan_concepts
                    )
                    # An edge referencing a now-deleted concept must go too,
                    # even if the edge's own chunk associations aren't all
                    # gone — one of its two endpoints no longer exists.
                    self._conn.execute(
                        f"DELETE FROM concept_edges WHERE source_concept_id IN ({op}) "
                        f"OR target_concept_id IN ({op})",
                        orphan_concepts + orphan_concepts,
                    )

                self._conn.execute(
                    "DELETE FROM concept_edges WHERE edge_id NOT IN "
                    "(SELECT DISTINCT edge_id FROM concept_edge_chunks)"
                )
                # Junction rows for edges deleted above (either path) are
                # now orphaned themselves — clean them up so a future
                # all_edges() never sees a phantom weight from a stale row.
                self._conn.execute(
                    "DELETE FROM concept_edge_chunks WHERE edge_id NOT IN "
                    "(SELECT edge_id FROM concept_edges)"
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def remove_source(self, source_id: str) -> None:
        """Resolves every chunk_id currently referenced anywhere in this
        store that belongs to `source_id` — via the chunk_id prefix
        convention f"{source_id}:{chunk_index}" (this store never persisted
        source_id as its own column) — prunes them via remove_chunk_ids,
        and deletes this source's own extraction checkpoints from
        concept_runs (otherwise a re-ingest under the same source_id, rare
        but possible, would wrongly be skipped as already-done)."""
        prefix = f"{source_id}:"
        rows = self._conn.execute(
            "SELECT DISTINCT chunk_id FROM concept_chunks "
            "UNION SELECT DISTINCT chunk_id FROM concept_edge_chunks"
        ).fetchall()
        matching = [r[0] for r in rows if r[0].startswith(prefix)]
        self.remove_chunk_ids(matching)

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM concept_runs WHERE source_id = ?", (source_id,))
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def prune_orphan_summaries(self) -> None:
        """Deletes community_summaries rows for communities no longer
        present in concept_communities — closes the leak where
        set_communities() replaces assignments but never garbage-collects
        the previous run's now-vanished communities' summaries. Called at
        the end of ragcore.concepts.communities.detect_communities()."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "DELETE FROM community_summaries WHERE community_id NOT IN "
                    "(SELECT DISTINCT community_id FROM concept_communities)"
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

    # -- internals -----------------------------------------------------

    def _chunk_ids_for_concepts(self, concept_ids: Sequence[str]) -> dict[str, list[str]]:
        concept_ids = list(dict.fromkeys(concept_ids))  # de-dupe, preserve order
        if not concept_ids:
            return {}
        placeholders = ",".join("?" * len(concept_ids))
        rows = self._conn.execute(
            f"SELECT concept_id, chunk_id FROM concept_chunks WHERE concept_id IN ({placeholders})",
            concept_ids,
        ).fetchall()
        out: dict[str, list[str]] = {}
        for concept_id, chunk_id in rows:
            out.setdefault(concept_id, []).append(chunk_id)
        return out

    def _chunk_ids_for_edges(self, edge_ids: Sequence[str]) -> dict[str, list[str]]:
        edge_ids = list(dict.fromkeys(edge_ids))
        if not edge_ids:
            return {}
        placeholders = ",".join("?" * len(edge_ids))
        rows = self._conn.execute(
            f"SELECT edge_id, chunk_id FROM concept_edge_chunks WHERE edge_id IN ({placeholders})",
            edge_ids,
        ).fetchall()
        out: dict[str, list[str]] = {}
        for edge_id, chunk_id in rows:
            out.setdefault(edge_id, []).append(chunk_id)
        return out
