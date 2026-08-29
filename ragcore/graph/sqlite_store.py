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
from pathlib import Path
from typing import Sequence

from ragcore.graph.types import Entity, Relation


def _merge_extra(old: dict, new: dict) -> dict:
    merged = dict(old)
    merged.update(new)
    old_aliases = set(old.get("aliases") or [])
    new_aliases = set(new.get("aliases") or [])
    if old_aliases or new_aliases:
        merged["aliases"] = sorted(old_aliases | new_aliases)
    return merged


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
    """Default GraphStore: one SQLite file, three-and-a-bit tables."""

    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- writes (merge, not replace) --------------------------------------

    def add_entities(self, entities: Sequence[Entity]) -> None:
        with self._conn:
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
                    merged = _merge_extra(json.loads(extra_json), e.extra)
                    self._conn.execute(
                        "UPDATE entities SET name = ?, extra_json = ? WHERE entity_id = ?",
                        (e.name, json.dumps(merged), e.entity_id),
                    )
                for chunk_id in e.source_chunk_ids:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO entity_chunks (entity_id, chunk_id) VALUES (?, ?)",
                        (e.entity_id, chunk_id),
                    )

    def add_relations(self, relations: Sequence[Relation]) -> None:
        with self._conn:
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
                    merged = _merge_extra(json.loads(extra_json), r.extra)
                    self._conn.execute(
                        "UPDATE relations SET extra_json = ? WHERE relation_id = ?",
                        (json.dumps(merged), r.relation_id),
                    )
                for chunk_id in r.source_chunk_ids:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO relation_chunks (relation_id, chunk_id) VALUES (?, ?)",
                        (r.relation_id, chunk_id),
                    )

    # -- reads ---------------------------------------------------------

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
                f"FROM relations WHERE source_entity_id IN ({placeholders}) "
                f"OR target_entity_id IN ({placeholders})"
            )
            params: list[str] = list(frontier) * 2
            if rel_types:
                type_placeholders = ",".join("?" * len(rel_types))
                sql += f" AND relation_type IN ({type_placeholders})"
                params += list(rel_types)

            rows = self._conn.execute(sql, params).fetchall()
            next_frontier: set[str] = set()
            for relation_id, source_id, target_id, relation_type, extra_json in rows:
                next_frontier.add(source_id)
                next_frontier.add(target_id)
                if relation_id in seen_relation_ids:
                    continue
                seen_relation_ids.add(relation_id)
                result.append(
                    Relation(
                        relation_id=relation_id,
                        source_entity_id=source_id,
                        target_entity_id=target_id,
                        relation_type=relation_type,
                        source_chunk_ids=tuple(self._chunk_ids_for_relation(relation_id)),
                        extra=json.loads(extra_json) if extra_json else {},
                    )
                )
            frontier = next_frontier

        return result

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

        out: dict[str, list[Entity]] = {}
        for chunk_id, entity_id, name, entity_type, extra_json in rows:
            entity = Entity(
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                source_chunk_ids=tuple(self._chunk_ids_for_entity(entity_id)),
                extra=json.loads(extra_json) if extra_json else {},
            )
            out.setdefault(chunk_id, []).append(entity)
        return out

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

    def entities_by_type(self, entity_type: str) -> list[Entity]:
        rows = self._conn.execute(
            "SELECT entity_id, name, extra_json FROM entities WHERE entity_type = ?",
            (entity_type,),
        ).fetchall()
        return [
            Entity(
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                source_chunk_ids=tuple(self._chunk_ids_for_entity(entity_id)),
                extra=json.loads(extra_json) if extra_json else {},
            )
            for entity_id, name, extra_json in rows
        ]

    # -- internals -----------------------------------------------------

    def _chunk_ids_for_entity(self, entity_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT chunk_id FROM entity_chunks WHERE entity_id = ?", (entity_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def _chunk_ids_for_relation(self, relation_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT chunk_id FROM relation_chunks WHERE relation_id = ?", (relation_id,)
        ).fetchall()
        return [r[0] for r in rows]
