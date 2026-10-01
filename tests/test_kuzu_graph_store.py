"""KuzuGraphStore, proven interchangeable with SqliteGraphStore: every
test in this file runs against both implementations via a parametrized
fixture (same "swap" pattern as test_store_swap.py for VectorStore) —
the strongest form of "the GraphStore protocol is genuinely swappable,"
docs/ARCHITECTURE.md Phase 4 gap list item 18.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("kuzu")

from ragcore.graph.kuzu_store import KuzuGraphStore
from ragcore.graph.sqlite_store import SqliteGraphStore
from ragcore.graph.types import Entity, Relation


@pytest.fixture(params=["sqlite", "kuzu"])
def store(request, tmp_path: Path):
    if request.param == "sqlite":
        return SqliteGraphStore(tmp_path / "graph.sqlite3")
    return KuzuGraphStore(tmp_path / "graph.kuzu")


def test_add_entities_merges_chunk_ids_and_aliases_on_repeat_write(store) -> None:
    store.add_entities(
        [Entity(entity_id="e1", name="Alice", entity_type="person", source_chunk_ids=("c1",), extra={"aliases": ["A"]})]
    )
    store.add_entities(
        [Entity(entity_id="e1", name="Alice", entity_type="person", source_chunk_ids=("c2",), extra={"aliases": ["Ally"]})]
    )

    [entity] = store.entities_by_type("person")
    assert entity.entity_id == "e1"
    assert set(entity.source_chunk_ids) == {"c1", "c2"}
    assert entity.extra["aliases"] == ["A", "Ally"]


def test_add_relations_merges_chunk_ids_on_repeat_write(store) -> None:
    store.add_entities(
        [
            Entity(entity_id="e1", name="A", entity_type="person", source_chunk_ids=("c1",), extra={}),
            Entity(entity_id="e2", name="B", entity_type="person", source_chunk_ids=("c1",), extra={}),
        ]
    )
    store.add_relations(
        [Relation(relation_id="r1", source_entity_id="e1", target_entity_id="e2", relation_type="knows", source_chunk_ids=("c1",), extra={})]
    )
    store.add_relations(
        [Relation(relation_id="r1", source_entity_id="e1", target_entity_id="e2", relation_type="knows", source_chunk_ids=("c2",), extra={})]
    )

    [relation] = store.neighbors(["e1"], hops=1)
    assert relation.relation_id == "r1"
    assert set(relation.source_chunk_ids) == {"c1", "c2"}


def test_neighbors_one_hop(store) -> None:
    store.add_entities(
        [Entity(entity_id=f"e{i}", name=f"E{i}", entity_type="person", source_chunk_ids=(f"c{i}",), extra={}) for i in range(3)]
    )
    store.add_relations(
        [
            Relation(relation_id="r01", source_entity_id="e0", target_entity_id="e1", relation_type="knows", source_chunk_ids=("c0",), extra={}),
            Relation(relation_id="r12", source_entity_id="e1", target_entity_id="e2", relation_type="owns", source_chunk_ids=("c1",), extra={}),
        ]
    )

    [relation] = store.neighbors(["e0"], hops=1)
    assert relation.relation_id == "r01"


def test_neighbors_two_hops_reaches_further(store) -> None:
    store.add_entities(
        [Entity(entity_id=f"e{i}", name=f"E{i}", entity_type="person", source_chunk_ids=(f"c{i}",), extra={}) for i in range(3)]
    )
    store.add_relations(
        [
            Relation(relation_id="r01", source_entity_id="e0", target_entity_id="e1", relation_type="knows", source_chunk_ids=("c0",), extra={}),
            Relation(relation_id="r12", source_entity_id="e1", target_entity_id="e2", relation_type="owns", source_chunk_ids=("c1",), extra={}),
        ]
    )

    one_hop_ids = {r.relation_id for r in store.neighbors(["e0"], hops=1)}
    two_hop_ids = {r.relation_id for r in store.neighbors(["e0"], hops=2)}

    assert one_hop_ids == {"r01"}
    assert two_hop_ids == {"r01", "r12"}


def test_neighbors_filters_by_relation_type(store) -> None:
    store.add_entities(
        [Entity(entity_id=f"e{i}", name=f"E{i}", entity_type="person", source_chunk_ids=(f"c{i}",), extra={}) for i in range(2)]
    )
    store.add_relations(
        [
            Relation(relation_id="r_knows", source_entity_id="e0", target_entity_id="e1", relation_type="knows", source_chunk_ids=("c0",), extra={}),
            Relation(relation_id="r_owns", source_entity_id="e0", target_entity_id="e1", relation_type="owns", source_chunk_ids=("c0",), extra={}),
        ]
    )

    knows_only = store.neighbors(["e0"], hops=1, rel_types=["knows"])
    assert [r.relation_id for r in knows_only] == ["r_knows"]


def test_entities_for_chunks(store) -> None:
    store.add_entities(
        [
            Entity(entity_id="e1", name="A", entity_type="person", source_chunk_ids=("c1", "c2"), extra={}),
            Entity(entity_id="e2", name="B", entity_type="person", source_chunk_ids=("c2",), extra={}),
        ]
    )

    result = store.entities_for_chunks(["c1", "c2"])

    assert {e.entity_id for e in result["c1"]} == {"e1"}
    assert {e.entity_id for e in result["c2"]} == {"e1", "e2"}


def test_chunks_for_entities(store) -> None:
    store.add_entities(
        [Entity(entity_id="e1", name="A", entity_type="person", source_chunk_ids=("c1", "c2"), extra={})]
    )
    result = store.chunks_for_entities(["e1"])
    assert set(result["e1"]) == {"c1", "c2"}


def test_remove_chunk_ids_partial_keeps_the_entity_alive(store) -> None:
    store.add_entities(
        [Entity(entity_id="e1", name="A", entity_type="person", source_chunk_ids=("c1", "c2"), extra={})]
    )

    store.remove_chunk_ids(["c1"])

    assert store.chunks_for_entities(["e1"]) == {"e1": ["c2"]}


def test_remove_chunk_ids_full_deletes_the_entity_and_its_relations(store) -> None:
    store.add_entities(
        [
            Entity(entity_id="e1", name="A", entity_type="person", source_chunk_ids=("c1",), extra={}),
            Entity(entity_id="e2", name="B", entity_type="person", source_chunk_ids=("c1",), extra={}),
        ]
    )
    store.add_relations(
        [Relation(relation_id="r1", source_entity_id="e1", target_entity_id="e2", relation_type="knows", source_chunk_ids=("c1",), extra={})]
    )

    store.remove_chunk_ids(["c1"])

    assert store.entities_by_type("person") == []
    assert store.neighbors(["e1"], hops=1) == []


def test_remove_source_prunes_by_chunk_id_prefix(store) -> None:
    store.add_entities(
        [
            Entity(entity_id="e1", name="A", entity_type="person", source_chunk_ids=("src1:0",), extra={}),
            Entity(entity_id="e2", name="B", entity_type="person", source_chunk_ids=("src2:0",), extra={}),
        ]
    )

    store.remove_source("src1")

    remaining = {e.entity_id for e in store.entities_by_type("person")}
    assert remaining == {"e2"}


def test_a_relation_can_reference_an_entity_not_yet_added(store) -> None:
    """No FK enforcement, same as SqliteGraphStore — a bare placeholder
    entity is created if add_relations() runs before add_entities()."""
    store.add_relations(
        [Relation(relation_id="r1", source_entity_id="new1", target_entity_id="new2", relation_type="knows", source_chunk_ids=("c1",), extra={})]
    )

    [relation] = store.neighbors(["new1"], hops=1)
    assert relation.target_entity_id == "new2"
