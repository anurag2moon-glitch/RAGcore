"""GraphStore implementation backed by stdlib sqlite3 — embedded and
zero-ops, the same rationale that chose LanceDB over a server-mode store
for vectors. No new dependency, no new pyproject.toml extra.

`add_entities`/`add_relations` merge into an existing row rather than
replacing it, when the (content-addressed) id already exists: source_chunk
ids and aliases accumulate as the same fact gets restated across chapters,
rather than the latest write clobbering earlier provenance. This is the one
place graph writes differ from VectorStore.upsert()'s replace-by-id
semantics — see the design doc's GraphRAG addendum.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Sequence

from ragcore.errors import wrap_backend_errors
from ragcore.graph.merge import merge_extra
from ragcore.graph.types import Entity, Relation

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    extra_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

CREATE TABLE IF NOT EXISTS entity_chunks (
    entity_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (entity_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_entity_chunks_chunk ON entity_chunks(chunk_id);

CREATE TABLE IF NOT EXISTS relations (
    relation_id TEXT PRIMARY KEY,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    extra_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity_id);

CREATE TABLE IF NOT EXISTS relation_chunks (
    relation_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (relation_id, chunk_id)
);
"""


class SqliteGraphStore:
    """Default GraphStore: one SQLite file, three-and-a-bit tables.

    Concurrency: `add_entities`/`add_relations` do a read-then-write merge
    (see module docstring), which is a lost-update race if two writers'
    reads interleave before either writes — real under concurrent ingestion,
    where several books' chapters resolve into the same entity at once.
    Guarded two ways: `check_same_thread=False` plus a `threading.Lock`
    serializes concurrent callers that share *one* `SqliteGraphStore`
    instance (the common case — one graph store, many ingestion threads);
    `isolation_level=None` (autocommit) with an explicit `BEGIN IMMEDIATE`
    around each merge acquires SQLite's write lock before the read, so
    concurrent *separate* connections to the same file (e.g. one per
    process) still serialize correctly rather than each reading a stale
    "not present yet" and clobbering the other's insert. `busy_timeout`
    makes a writer wait for that lock instead of raising immediately.
    """

    @wrap_backend_errors
    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)

    # -- writes (merge, not replace) --------------------------------------

    @wrap_backend_errors
    def add_entities(self, entities: Sequence[Entity]) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for e in entities:
                    row = self._conn.execute(
                        "SELECT name, extra_json FROM entities WHERE entity_id = ?",
                        (e.entity_id,),
                    ).fetchone()
                    if row is None:
                        self._conn.execute(
                            "INSERT INTO entities (entity_id, name, entity_type, extra_json) "
                            "VALUES (?, ?, ?, ?)",
                            (e.entity_id, e.name, e.entity_type, json.dumps(e.extra)),
                        )
                    else:
                        _name, extra_json = row
                        merged = merge_extra(json.loads(extra_json), e.extra)
                        self._conn.execute(
                            "UPDATE entities SET name = ?, extra_json = ? WHERE entity_id = ?",
                            (e.name, json.dumps(merged), e.entity_id),
                        )
                    for chunk_id in e.source_chunk_ids:
                        self._conn.execute(
                            "INSERT OR IGNORE INTO entity_chunks (entity_id, chunk_id) VALUES (?, ?)",
                            (e.entity_id, chunk_id),
                        )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def add_relations(self, relations: Sequence[Relation]) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for r in relations:
                    row = self._conn.execute(
                        "SELECT extra_json FROM relations WHERE relation_id = ?",
                        (r.relation_id,),
                    ).fetchone()
                    if row is None:
                        self._conn.execute(
                            "INSERT INTO relations "
                            "(relation_id, source_entity_id, target_entity_id, relation_type, extra_json) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (
                                r.relation_id,
                                r.source_entity_id,
                                r.target_entity_id,
                                r.relation_type,
                                json.dumps(r.extra),
                            ),
                        )
                    else:
                        (extra_json,) = row
                        merged = merge_extra(json.loads(extra_json), r.extra)
                        self._conn.execute(
                            "UPDATE relations SET extra_json = ? WHERE relation_id = ?",
                            (json.dumps(merged), r.relation_id),
                        )
                    for chunk_id in r.source_chunk_ids:
                        self._conn.execute(
                            "INSERT OR IGNORE INTO relation_chunks (relation_id, chunk_id) VALUES (?, ?)",
                            (r.relation_id, chunk_id),
                        )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    # -- reads ---------------------------------------------------------

    @wrap_backend_errors
    def neighbors(
        self,
        entity_ids: Sequence[str],
        *,
        hops: int = 1,
        rel_types: Sequence[str] | None = None,
    ) -> list[Relation]:
        frontier = set(entity_ids)
        seen_relation_ids: set[str] = set()
        result: list[Relation] = []

        for _ in range(hops):
            if not frontier:
                break
            placeholders = ",".join("?" * len(frontier))
            sql = (
                f"SELECT relation_id, source_entity_id, target_entity_id, relation_type, extra_json "
                f"FROM relations WHERE (source_entity_id IN ({placeholders}) "
                f"OR target_entity_id IN ({placeholders}))"
            )
            params: list[str] = list(frontier) * 2
            if rel_types:
                # Parenthesized above so this AND constrains the whole
                # source-OR-target match, not just the target side (SQL's
                # default AND/OR precedence would otherwise silently let a
                # source-side hit through regardless of its relation_type).
                type_placeholders = ",".join("?" * len(rel_types))
                sql += f" AND relation_type IN ({type_placeholders})"
                params += list(rel_types)

            rows = self._conn.execute(sql, params).fetchall()
            hop_relations: list[tuple[str, str, str, str, str]] = []
            next_frontier: set[str] = set()
            for relation_id, source_id, target_id, relation_type, extra_json in rows:
                next_frontier.add(source_id)
                next_frontier.add(target_id)
                if relation_id in seen_relation_ids:
                    continue
                seen_relation_ids.add(relation_id)
                hop_relations.append((relation_id, source_id, target_id, relation_type, extra_json))

            # One batched fetch of provenance chunk ids for every relation
            # discovered this hop, instead of one query per relation.
            chunk_ids_by_relation = self._chunk_ids_for_relations(
                [r[0] for r in hop_relations]
            )
            for relation_id, source_id, target_id, relation_type, extra_json in hop_relations:
                result.append(
                    Relation(
                        relation_id=relation_id,
                        source_entity_id=source_id,
                        target_entity_id=target_id,
                        relation_type=relation_type,
                        source_chunk_ids=tuple(chunk_ids_by_relation.get(relation_id, [])),
                        extra=json.loads(extra_json) if extra_json else {},
                    )
                )
            frontier = next_frontier

        return result

    @wrap_backend_errors
    def entities_for_chunks(self, chunk_ids: Sequence[str]) -> dict[str, list[Entity]]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT ec.chunk_id, e.entity_id, e.name, e.entity_type, e.extra_json "
            f"FROM entity_chunks ec JOIN entities e ON ec.entity_id = e.entity_id "
            f"WHERE ec.chunk_id IN ({placeholders})",
            list(chunk_ids),
        ).fetchall()

        # One batched fetch of every matched entity's chunk ids, instead of
        # one query per (chunk, entity) row.
        chunk_ids_by_entity = self._chunk_ids_for_entities({r[1] for r in rows})

        out: dict[str, list[Entity]] = {}
        for chunk_id, entity_id, name, entity_type, extra_json in rows:
            entity = Entity(
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                source_chunk_ids=tuple(chunk_ids_by_entity.get(entity_id, [])),
                extra=json.loads(extra_json) if extra_json else {},
            )
            out.setdefault(chunk_id, []).append(entity)
        return out

    @wrap_backend_errors
    def chunks_for_entities(self, entity_ids: Sequence[str]) -> dict[str, list[str]]:
        if not entity_ids:
            return {}
        placeholders = ",".join("?" * len(entity_ids))
        rows = self._conn.execute(
            f"SELECT entity_id, chunk_id FROM entity_chunks WHERE entity_id IN ({placeholders})",
            list(entity_ids),
        ).fetchall()
        out: dict[str, list[str]] = {}
        for entity_id, chunk_id in rows:
            out.setdefault(entity_id, []).append(chunk_id)
        return out

    @wrap_backend_errors
    def entities_by_type(self, entity_type: str) -> list[Entity]:
        rows = self._conn.execute(
            "SELECT entity_id, name, extra_json FROM entities WHERE entity_type = ?",
            (entity_type,),
        ).fetchall()
        # One batched fetch of every entity's chunk ids, instead of one
        # query per entity — this is called once per candidate during
        # build_graph() resolution, so N+1 here means N+1 queries per
        # candidate, per chapter, across the whole library.
        chunk_ids_by_entity = self._chunk_ids_for_entities([r[0] for r in rows])
        return [
            Entity(
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                source_chunk_ids=tuple(chunk_ids_by_entity.get(entity_id, [])),
                extra=json.loads(extra_json) if extra_json else {},
            )
            for entity_id, name, extra_json in rows
        ]

    # -- removal (item 7: incremental ingestion) -----------------------

    @wrap_backend_errors
    def remove_chunk_ids(self, chunk_ids: Sequence[str]) -> None:
        """Prunes the given chunk_ids from every entity's/relation's
        provenance, deleting any entity or relation left with zero
        remaining chunk associations. Used by
        ragcore.versioning.sync.sync_corpus when a source is removed or
        changed (a changed file is re-ingested under a brand-new
        content-addressed source_id, so its old chunks must be explicitly
        pruned here — nothing else in this store would ever notice they're
        gone otherwise)."""
        if not chunk_ids:
            return
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                placeholders = ",".join("?" * len(chunk_ids))
                ids = list(chunk_ids)
                self._conn.execute(
                    f"DELETE FROM entity_chunks WHERE chunk_id IN ({placeholders})", ids
                )
                self._conn.execute(
                    f"DELETE FROM relation_chunks WHERE chunk_id IN ({placeholders})", ids
                )
                self._conn.execute(
                    "DELETE FROM entities WHERE entity_id NOT IN "
                    "(SELECT DISTINCT entity_id FROM entity_chunks)"
                )
                self._conn.execute(
                    "DELETE FROM relations WHERE relation_id NOT IN "
                    "(SELECT DISTINCT relation_id FROM relation_chunks)"
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
        convention f"{source_id}:{chunk_index}" (ragcore.types.Chunk.chunk_id
        is content-addressed this way; this store never persisted source_id
        as its own column, so the prefix is the only way to recover it) —
        and prunes them via remove_chunk_ids."""
        prefix = f"{source_id}:"
        rows = self._conn.execute(
            "SELECT DISTINCT chunk_id FROM entity_chunks "
            "UNION SELECT DISTINCT chunk_id FROM relation_chunks"
        ).fetchall()
        matching = [r[0] for r in rows if r[0].startswith(prefix)]
        self.remove_chunk_ids(matching)

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
        in place from a previously-taken backup file, via the same stdlib
        sqlite3 backup API used in reverse — verified safe to call against
        a live, open destination connection (that's what the backup API is
        for), so no external close/reopen of this store's connection or
        file is needed. Used by ragcore.versioning.snapshot.rollback."""
        with self._lock:
            source = sqlite3.connect(str(path))
            try:
                source.backup(self._conn)
            finally:
                source.close()

    # -- internals -----------------------------------------------------

    def _chunk_ids_for_entities(self, entity_ids: Sequence[str]) -> dict[str, list[str]]:
        entity_ids = list(dict.fromkeys(entity_ids))  # de-dupe, preserve order
        if not entity_ids:
            return {}
        placeholders = ",".join("?" * len(entity_ids))
        rows = self._conn.execute(
            f"SELECT entity_id, chunk_id FROM entity_chunks WHERE entity_id IN ({placeholders})",
            entity_ids,
        ).fetchall()
        out: dict[str, list[str]] = {}
        for entity_id, chunk_id in rows:
            out.setdefault(entity_id, []).append(chunk_id)
        return out

    def _chunk_ids_for_relations(self, relation_ids: Sequence[str]) -> dict[str, list[str]]:
        relation_ids = list(dict.fromkeys(relation_ids))
        if not relation_ids:
            return {}
        placeholders = ",".join("?" * len(relation_ids))
        rows = self._conn.execute(
            f"SELECT relation_id, chunk_id FROM relation_chunks WHERE relation_id IN ({placeholders})",
            relation_ids,
        ).fetchall()
        out: dict[str, list[str]] = {}
        for relation_id, chunk_id in rows:
            out.setdefault(relation_id, []).append(chunk_id)
        return out
