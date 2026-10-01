"""GraphStore implementation backed by Kuzu — embedded, zero-ops, the
same rationale that chose SQLite/LanceDB for the other default stores
(docs/ARCHITECTURE.md Phase 4 gap list item 18: a graph-store adapter
behind the existing GraphStore protocol, SQLite-only until now).

Schema is a genuine property graph (one Entity node table, one RELATES
rel table) rather than SqliteGraphStore's relational-tables-standing-in-
for-a-graph. `source_chunk_ids` is a native `STRING[]` list property on
both tables (Kuzu supports list columns directly), so there's no
separate join table the way SqliteGraphStore needs `entity_chunks`/
`relation_chunks` — a list union is one `list_distinct(old + new)` Cypher
expression.

`neighbors()` still runs its own hop-by-hop loop in Python, mirroring
SqliteGraphStore's loop exactly, rather than one variable-length Cypher
path query: a `*1..N` pattern match returns reachable *nodes*, not the
individual `Relation` objects (with their own relation_id/relation_type/
extra/provenance) this protocol's `neighbors()` contract returns. An
undirected per-hop MATCH also returns the same edge once per traversal
direction when *both* of its endpoints are already in the frontier
(confirmed empirically against this Kuzu version) — handled for free by
the same `seen_relation_ids` dedup-by-relation_id set
SqliteGraphStore's loop already uses, not a new concern this store
introduces.

`source_entity_id`/`target_entity_id` are stored as explicit REL
properties, not read back via a structural "which end is FROM/TO"
function: this Kuzu version has no `startNode()`/`endNode()` (confirmed:
"function STARTNODE does not exist"), and an undirected MATCH's own
pattern-variable bindings don't reliably preserve which side was
originally added as source vs. target. Storing them as plain properties
— exactly what SqliteGraphStore already does with real columns — avoids
needing either.

`extra_json`'s semantic dict merge (`ragcore.graph.merge.merge_extra` —
dict-update plus a list-union on specific keys) is done in Python, not
in Cypher: a `MERGE`'s `ON MATCH SET` can union a *list* property
server-side (used for `source_chunk_ids`), but there's no server-side
equivalent for a JSON string's semantic merge, so this store reads the
existing row first — the same read-then-merge-then-write pattern
SqliteGraphStore already uses for the identical reason.

**Concurrency, stated plainly**: unlike SqliteGraphStore (explicitly
hardened and tested for concurrent threads *and* concurrent separate
connections/processes to one file — see its own module docstring), this
store's thread-safety is a `threading.Lock()` around each write,
guarding concurrent callers that share *one* `KuzuGraphStore` instance
within *one process* — the same guarantee level, no more. Two
independent processes opening the same database file concurrently was
not characterized deeply enough in this pass to make the same
multi-process claim; don't rely on it until it's been verified.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Sequence

from ragcore.errors import StoreError, wrap_backend_errors
from ragcore.graph.merge import merge_extra
from ragcore.graph.types import Entity, Relation

try:
    import kuzu
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    kuzu = None
else:
    _IMPORT_ERROR = None

_SCHEMA = [
    (
        "CREATE NODE TABLE IF NOT EXISTS Entity("
        "entity_id STRING, name STRING, entity_type STRING, extra_json STRING, "
        "source_chunk_ids STRING[], PRIMARY KEY(entity_id))"
    ),
    (
        "CREATE REL TABLE IF NOT EXISTS RELATES(FROM Entity TO Entity, "
        "relation_id STRING, source_entity_id STRING, target_entity_id STRING, "
        "relation_type STRING, extra_json STRING, source_chunk_ids STRING[])"
    ),
]


def _row_to_entity(entity_id: str, name: str, entity_type: str, extra_json: str, chunk_ids: list[str]) -> Entity:
    return Entity(
        entity_id=entity_id,
        name=name,
        entity_type=entity_type,
        source_chunk_ids=tuple(chunk_ids),
        extra=json.loads(extra_json) if extra_json else {},
    )


class KuzuGraphStore:
    """GraphStore backed by an embedded Kuzu database file."""

    @wrap_backend_errors
    def __init__(self, path: str | Path) -> None:
        if kuzu is None:
            raise StoreError(
                "Kuzu graph storage requires the 'kuzu' extra: install ragcore[kuzu]"
            ) from _IMPORT_ERROR

        self._lock = threading.Lock()
        self._db = kuzu.Database(str(path))
        self._conn = kuzu.Connection(self._db)
        for statement in _SCHEMA:
            self._conn.execute(statement)

    # -- writes (merge, not replace) --------------------------------------

    @wrap_backend_errors
    def add_entities(self, entities: Sequence[Entity]) -> None:
        with self._lock:
            self._conn.execute("BEGIN TRANSACTION")
            try:
                for e in entities:
                    existing = self._conn.execute(
                        "MATCH (n:Entity {entity_id: $id}) "
                        "RETURN n.extra_json, n.source_chunk_ids",
                        {"id": e.entity_id},
                    )
                    if existing.has_next():
                        extra_json, old_chunk_ids = existing.get_next()
                        merged_extra = merge_extra(json.loads(extra_json), e.extra)
                        merged_chunk_ids = sorted(set(old_chunk_ids) | set(e.source_chunk_ids))
                    else:
                        merged_extra = e.extra
                        merged_chunk_ids = sorted(set(e.source_chunk_ids))
                    self._conn.execute(
                        "MERGE (n:Entity {entity_id: $id}) "
                        "ON CREATE SET n.name = $name, n.entity_type = $type, "
                        "n.extra_json = $extra, n.source_chunk_ids = $chunks "
                        "ON MATCH SET n.name = $name, n.extra_json = $extra, "
                        "n.source_chunk_ids = $chunks",
                        {
                            "id": e.entity_id,
                            "name": e.name,
                            "type": e.entity_type,
                            "extra": json.dumps(merged_extra),
                            "chunks": merged_chunk_ids,
                        },
                    )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @wrap_backend_errors
    def add_relations(self, relations: Sequence[Relation]) -> None:
        with self._lock:
            self._conn.execute("BEGIN TRANSACTION")
            try:
                for r in relations:
                    # Permissive like SqliteGraphStore: a relation can
                    # reference an entity add_entities() hasn't been
                    # called for yet (no FK enforcement either way) — a
                    # bare placeholder node is created if missing.
                    self._conn.execute(
                        "MERGE (n:Entity {entity_id: $id}) "
                        "ON CREATE SET n.name = $id, n.entity_type = '', "
                        "n.extra_json = '{}', n.source_chunk_ids = []",
                        {"id": r.source_entity_id},
                    )
                    self._conn.execute(
                        "MERGE (n:Entity {entity_id: $id}) "
                        "ON CREATE SET n.name = $id, n.entity_type = '', "
                        "n.extra_json = '{}', n.source_chunk_ids = []",
                        {"id": r.target_entity_id},
                    )

                    existing = self._conn.execute(
                        "MATCH (:Entity)-[rel:RELATES {relation_id: $id}]->(:Entity) "
                        "RETURN rel.extra_json, rel.source_chunk_ids",
                        {"id": r.relation_id},
                    )
                    if existing.has_next():
                        extra_json, old_chunk_ids = existing.get_next()
                        merged_extra = merge_extra(json.loads(extra_json), r.extra)
                        merged_chunk_ids = sorted(set(old_chunk_ids) | set(r.source_chunk_ids))
                    else:
                        merged_extra = r.extra
                        merged_chunk_ids = sorted(set(r.source_chunk_ids))

                    self._conn.execute(
                        "MATCH (a:Entity {entity_id: $src}), (b:Entity {entity_id: $tgt}) "
                        "MERGE (a)-[rel:RELATES {relation_id: $id}]->(b) "
                        "ON CREATE SET rel.source_entity_id = $src, rel.target_entity_id = $tgt, "
                        "rel.relation_type = $type, rel.extra_json = $extra, "
                        "rel.source_chunk_ids = $chunks "
                        "ON MATCH SET rel.extra_json = $extra, rel.source_chunk_ids = $chunks",
                        {
                            "id": r.relation_id,
                            "src": r.source_entity_id,
                            "tgt": r.target_entity_id,
                            "type": r.relation_type,
                            "extra": json.dumps(merged_extra),
                            "chunks": merged_chunk_ids,
                        },
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
            query = (
                "MATCH (n:Entity)-[r:RELATES]-(:Entity) WHERE n.entity_id IN $ids"
            )
            params: dict[str, Any] = {"ids": list(frontier)}
            if rel_types:
                query += " AND r.relation_type IN $types"
                params["types"] = list(rel_types)
            query += (
                " RETURN DISTINCT r.relation_id, r.source_entity_id, r.target_entity_id, "
                "r.relation_type, r.extra_json, r.source_chunk_ids"
            )

            rows = self._conn.execute(query, params)
            next_frontier: set[str] = set()
            while rows.has_next():
                relation_id, source_id, target_id, relation_type, extra_json, chunk_ids = rows.get_next()
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
                        source_chunk_ids=tuple(chunk_ids),
                        extra=json.loads(extra_json) if extra_json else {},
                    )
                )
            frontier = next_frontier

        return result

    @wrap_backend_errors
    def entities_for_chunks(self, chunk_ids: Sequence[str]) -> dict[str, list[Entity]]:
        if not chunk_ids:
            return {}
        out: dict[str, list[Entity]] = {}
        rows = self._conn.execute(
            "MATCH (n:Entity) RETURN n.entity_id, n.name, n.entity_type, "
            "n.extra_json, n.source_chunk_ids"
        )
        wanted = set(chunk_ids)
        while rows.has_next():
            entity_id, name, entity_type, extra_json, chunks = rows.get_next()
            hit_chunks = [c for c in chunks if c in wanted]
            if not hit_chunks:
                continue
            entity = _row_to_entity(entity_id, name, entity_type, extra_json, chunks)
            for chunk_id in hit_chunks:
                out.setdefault(chunk_id, []).append(entity)
        return out

    @wrap_backend_errors
    def chunks_for_entities(self, entity_ids: Sequence[str]) -> dict[str, list[str]]:
        if not entity_ids:
            return {}
        out: dict[str, list[str]] = {}
        rows = self._conn.execute(
            "MATCH (n:Entity) WHERE n.entity_id IN $ids RETURN n.entity_id, n.source_chunk_ids",
            {"ids": list(entity_ids)},
        )
        while rows.has_next():
            entity_id, chunks = rows.get_next()
            out[entity_id] = list(chunks)
        return out

    @wrap_backend_errors
    def entities_by_type(self, entity_type: str) -> list[Entity]:
        rows = self._conn.execute(
            "MATCH (n:Entity {entity_type: $type}) "
            "RETURN n.entity_id, n.name, n.extra_json, n.source_chunk_ids",
            {"type": entity_type},
        )
        out: list[Entity] = []
        while rows.has_next():
            entity_id, name, extra_json, chunks = rows.get_next()
            out.append(_row_to_entity(entity_id, name, entity_type, extra_json, chunks))
        return out

    # -- removal (incremental ingestion) -----------------------------------

    @wrap_backend_errors
    def remove_chunk_ids(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        to_remove = set(chunk_ids)
        with self._lock:
            self._conn.execute("BEGIN TRANSACTION")
            try:
                rel_rows = self._conn.execute(
                    "MATCH (:Entity)-[r:RELATES]->(:Entity) "
                    "RETURN r.relation_id, r.source_entity_id, r.target_entity_id, r.source_chunk_ids"
                )
                relations_to_delete: list[tuple[str, str, str]] = []
                relations_to_update: list[tuple[str, str, str, list[str]]] = []
                while rel_rows.has_next():
                    relation_id, src, tgt, chunks = rel_rows.get_next()
                    remaining = [c for c in chunks if c not in to_remove]
                    if remaining == list(chunks):
                        continue
                    if remaining:
                        relations_to_update.append((relation_id, src, tgt, remaining))
                    else:
                        relations_to_delete.append((relation_id, src, tgt))

                for relation_id, src, tgt, remaining in relations_to_update:
                    self._conn.execute(
                        "MATCH (a:Entity {entity_id: $src})-[r:RELATES {relation_id: $id}]->"
                        "(b:Entity {entity_id: $tgt}) SET r.source_chunk_ids = $chunks",
                        {"src": src, "tgt": tgt, "id": relation_id, "chunks": remaining},
                    )
                for relation_id, src, tgt in relations_to_delete:
                    self._conn.execute(
                        "MATCH (a:Entity {entity_id: $src})-[r:RELATES {relation_id: $id}]->"
                        "(b:Entity {entity_id: $tgt}) DELETE r",
                        {"src": src, "tgt": tgt, "id": relation_id},
                    )

                entity_rows = self._conn.execute(
                    "MATCH (n:Entity) RETURN n.entity_id, n.source_chunk_ids"
                )
                entities_to_delete: list[str] = []
                entities_to_update: list[tuple[str, list[str]]] = []
                while entity_rows.has_next():
                    entity_id, chunks = entity_rows.get_next()
                    remaining = [c for c in chunks if c not in to_remove]
                    if remaining == list(chunks):
                        continue
                    if remaining:
                        entities_to_update.append((entity_id, remaining))
                    else:
                        entities_to_delete.append(entity_id)

                for entity_id, remaining in entities_to_update:
                    self._conn.execute(
                        "MATCH (n:Entity {entity_id: $id}) SET n.source_chunk_ids = $chunks",
                        {"id": entity_id, "chunks": remaining},
                    )
                for entity_id in entities_to_delete:
                    self._conn.execute(
                        "MATCH (n:Entity {entity_id: $id}) DETACH DELETE n", {"id": entity_id}
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
        convention f"{source_id}:{chunk_index}", same as SqliteGraphStore
        — and prunes them via remove_chunk_ids."""
        prefix = f"{source_id}:"
        seen: set[str] = set()
        entity_rows = self._conn.execute("MATCH (n:Entity) RETURN n.source_chunk_ids")
        while entity_rows.has_next():
            (chunks,) = entity_rows.get_next()
            seen.update(c for c in chunks if c.startswith(prefix))
        rel_rows = self._conn.execute("MATCH (:Entity)-[r:RELATES]->(:Entity) RETURN r.source_chunk_ids")
        while rel_rows.has_next():
            (chunks,) = rel_rows.get_next()
            seen.update(c for c in chunks if c.startswith(prefix))
        self.remove_chunk_ids(sorted(seen))
