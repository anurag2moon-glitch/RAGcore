"""SqliteGraphStore.remove_chunk_ids/remove_source: pruning provenance and
deleting orphaned entities/relations, plus the online-backup helper — the
new removal capability item 7 (incremental ingestion) needs, since the
graph store has otherwise only ever been additive.
"""

from __future__ import annotations

from pathlib import Path

from ragcore.graph.sqlite_store import SqliteGraphStore
from ragcore.graph.types import Entity, Relation


def test_remove_source_prunes_provenance_but_keeps_entities_with_other_sources() -> None:
    store = SqliteGraphStore(":memory:")
    store.add_entities(
        [
            Entity(entity_id="entity:person:a", name="A", entity_type="person",
                   source_chunk_ids=("src1:0", "src2:0")),
            Entity(entity_id="entity:person:b", name="B", entity_type="person",
                   source_chunk_ids=("src2:0",)),
        ]
    )

    store.remove_source("src1")

    remaining = {e.entity_id for e in store.entities_by_type("person")}
    assert remaining == {"entity:person:a", "entity:person:b"}
    assert store.chunks_for_entities(["entity:person:a"])["entity:person:a"] == ["src2:0"]


def test_remove_source_deletes_an_entity_left_with_zero_chunks() -> None:
    store = SqliteGraphStore(":memory:")
    store.add_entities(
        [Entity(entity_id="entity:person:x", name="X", entity_type="person",
                source_chunk_ids=("src1:0", "src1:1"))]
    )

    store.remove_source("src1")

    assert store.entities_by_type("person") == []


def test_remove_source_deletes_a_relation_left_with_zero_chunks() -> None:
    store = SqliteGraphStore(":memory:")
    store.add_entities(
        [
            Entity(entity_id="entity:person:a", name="A", entity_type="person",
                   source_chunk_ids=("src1:0", "src2:0")),
            Entity(entity_id="entity:person:b", name="B", entity_type="person",
                   source_chunk_ids=("src2:0",)),
        ]
    )
    store.add_relations(
        [
            Relation(relation_id="relation:knows:entity:person:a:entity:person:b",
                      source_entity_id="entity:person:a", target_entity_id="entity:person:b",
                      relation_type="knows", source_chunk_ids=("src1:0",))
        ]
    )

    store.remove_source("src1")

    assert store.neighbors(["entity:person:a", "entity:person:b"]) == []


def test_remove_chunk_ids_on_empty_list_is_a_no_op() -> None:
    store = SqliteGraphStore(":memory:")
    store.add_entities(
        [Entity(entity_id="entity:person:a", name="A", entity_type="person", source_chunk_ids=("s:0",))]
    )

    store.remove_chunk_ids([])

    assert len(store.entities_by_type("person")) == 1


def test_backup_to_produces_a_restorable_copy(tmp_path: Path) -> None:
    store = SqliteGraphStore(":memory:")
    store.add_entities(
        [Entity(entity_id="entity:person:z", name="Z", entity_type="person", source_chunk_ids=("s:0",))]
    )

    backup_path = tmp_path / "backup.sqlite3"
    store.backup_to(backup_path)

    restored = SqliteGraphStore(backup_path)
    assert [e.entity_id for e in restored.entities_by_type("person")] == ["entity:person:z"]
