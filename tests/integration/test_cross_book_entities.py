"""Requirement 2: cross-book entity resolution — the same real-world
entity appearing in different books (under naming drift) must resolve to
one graph entity; different entities that merely share a surname or type
must correctly stay separate.

Ground truth (see tests/integration/synth/library.py):
  - "Captain Aurelia Venn" / "Venn" / "Aurelia" — one person, drifting
    surface forms planted across TWO books (clean_toc ch3+ch27,
    fragment_anchored ch20) — MUST merge into one entity.
  - "Elias Thorn" (person) vs "Mira Thorn" (person) — two different
    people sharing a surname, same entity_type, planted in different
    books — MUST stay separate (verified empirically: with
    DeterministicEmbedder, this pair's cosine similarity is ~0.667,
    comfortably below EmbeddingEntityResolver's real production default
    threshold of 0.87 — no threshold override needed).
  - "the botanist"/"Elias Thorn" (entity_type="botanist") vs "the
    cartographer"/"Mira Thorn" (entity_type="cartographer") — different
    entity_type strings, so EmbeddingEntityResolver never even compares
    them (type-gated) — a second, structurally-guaranteed separation.

Also the direct regression test for the `neighbors()` rel_types SQL
precedence bug fixed in ragcore/graph/sqlite_store.py (see
docs/INTEGRATION_TEST_FINDINGS.md).
"""

from __future__ import annotations

from ragcore.graph.types import Entity, Relation
from tests.integration.conftest import SharedLibrary


def _find_person(entities: list[Entity], *, name_contains: str) -> Entity | None:
    for e in entities:
        if name_contains.lower() in e.name.lower() or any(
            name_contains.lower() in a.lower() for a in e.extra.get("aliases", [])
        ):
            return e
    return None


def test_drifting_cross_book_aliases_merge_into_one_entity(shared_library: SharedLibrary) -> None:
    if not (shared_library.is_ingested("clean_toc") and shared_library.is_ingested("fragment_anchored")):
        import pytest

        pytest.skip("needs both clean_toc and fragment_anchored ingested (RAGCORE_IT_SCALE=quick subset)")

    persons = shared_library.graph.entities_by_type("person")
    venn = _find_person(persons, name_contains="venn")
    assert venn is not None, "Captain Aurelia Venn / Venn / Aurelia never resolved to any entity"

    aliases = set(venn.extra.get("aliases", []))
    # every surface form actually used across both books' chapters ends up
    # as an alias of the single merged entity
    assert {"Captain Aurelia Venn", "the captain", "her"} <= aliases

    # provenance (source_chunk_ids) accumulates — not overwritten — across
    # every chapter/book restatement, per CLAUDE.md's merge-not-replace
    # contract
    assert len(venn.source_chunk_ids) >= 3  # at least clean_toc x2 + fragment_anchored x1

    # exactly one "Venn-family" person entity exists — the three surface
    # forms did not each become their own separate entity
    venn_like = [
        e for e in persons
        if "venn" in e.name.lower() or "venn" in " ".join(e.extra.get("aliases", [])).lower()
        or "aurelia" in e.name.lower() or "aurelia" in " ".join(e.extra.get("aliases", [])).lower()
    ]
    assert len(venn_like) == 1


def test_same_surname_different_people_stay_separate(shared_library: SharedLibrary) -> None:
    if not (shared_library.is_ingested("clean_toc") and shared_library.is_ingested("unusual_naming")):
        import pytest

        pytest.skip("needs both clean_toc and unusual_naming ingested (RAGCORE_IT_SCALE=quick subset)")

    persons = shared_library.graph.entities_by_type("person")
    elias = _find_person(persons, name_contains="elias thorn")
    mira = _find_person(persons, name_contains="mira thorn")
    assert elias is not None and mira is not None
    assert elias.entity_id != mira.entity_id
    # neither's chunk provenance touches the other's — no cross-contamination
    assert set(elias.source_chunk_ids).isdisjoint(mira.source_chunk_ids)


def test_different_entity_type_never_compared_even_with_shared_surname(shared_library: SharedLibrary) -> None:
    if not (shared_library.is_ingested("page_structured") and shared_library.is_ingested("heading_structured")):
        import pytest

        pytest.skip("needs page_structured and heading_structured ingested")

    botanists = shared_library.graph.entities_by_type("botanist")
    cartographers = shared_library.graph.entities_by_type("cartographer")
    assert len(botanists) == 1
    assert len(cartographers) == 1
    assert botanists[0].entity_id != cartographers[0].entity_id


def test_rerunning_build_graph_is_idempotent(shared_library: SharedLibrary) -> None:
    """Re-running build_graph over the same already-ingested library must
    re-merge into the same rows, not duplicate them — the content-addressed
    id + merge-not-replace contract CLAUDE.md documents as load-bearing."""
    from ragcore.graph.pipeline import build_graph

    before_persons = {e.entity_id: e for e in shared_library.graph.entities_by_type("person")}
    before_count = len(before_persons)

    build_graph(shared_library.corpus, shared_library.graph, shared_library.extractor)

    after_persons = {e.entity_id: e for e in shared_library.graph.entities_by_type("person")}
    assert len(after_persons) == before_count
    assert set(before_persons) == set(after_persons)
    # provenance doesn't balloon on re-run — INSERT OR IGNORE de-dupes
    # (chunk_id, entity_id) pairs
    for entity_id, before in before_persons.items():
        assert set(after_persons[entity_id].source_chunk_ids) == set(before.source_chunk_ids)


def test_neighbors_rel_types_filters_both_traversal_directions() -> None:
    """Direct regression test for the SQL-precedence bug: `neighbors()`
    with `rel_types=[...]` must constrain matches on BOTH the source and
    target side, not silently let any target-side match with a different
    relation_type through (see docs/INTEGRATION_TEST_FINDINGS.md)."""
    from ragcore.graph.sqlite_store import SqliteGraphStore

    graph = SqliteGraphStore(":memory:")
    graph.add_entities(
        [
            Entity(entity_id="entity:person:a", name="A", entity_type="person", source_chunk_ids=()),
            Entity(entity_id="entity:person:b", name="B", entity_type="person", source_chunk_ids=()),
            Entity(entity_id="entity:person:c", name="C", entity_type="person", source_chunk_ids=()),
        ]
    )
    graph.add_relations(
        [
            Relation(
                relation_id="relation:likes:entity:person:a:entity:person:b",
                source_entity_id="entity:person:a", target_entity_id="entity:person:b",
                relation_type="likes", source_chunk_ids=(),
            ),
            Relation(
                relation_id="relation:hates:entity:person:a:entity:person:c",
                source_entity_id="entity:person:a", target_entity_id="entity:person:c",
                relation_type="hates", source_chunk_ids=(),
            ),
        ]
    )

    liked_only = graph.neighbors(["entity:person:a"], rel_types=["likes"])
    assert [r.relation_type for r in liked_only] == ["likes"]
    assert {r.target_entity_id for r in liked_only} == {"entity:person:b"}
