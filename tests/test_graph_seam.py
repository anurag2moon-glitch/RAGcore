"""Graph seam test: proves the integration points the future graph layer
needs are wired correctly *before* there is any graph implementation.
See design doc §4.
"""

from __future__ import annotations

from pathlib import Path

from ragcore.corpus import Corpus
from ragcore.graph import Entity, EntityExtractor, GraphContext, GraphStore, Relation
from tests.fakes import FakeEmbedder, InMemoryStore


def test_retrieved_graph_field_defaults_to_none(md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)

    results = corpus.retrieve("chunking", k=10)
    assert results
    assert all(r.graph is None for r in results)


def test_store_get_roundtrips_by_chunk_id(md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)

    results = corpus.retrieve("chunking", k=10)
    ids = [r.chunk.chunk_id for r in results]

    hydrated = corpus.get(ids)
    assert {c.chunk_id for c in hydrated} == set(ids)


def test_graph_protocols_are_structurally_satisfiable() -> None:
    """Real implementations exist now (SqliteGraphStore, ClaudeEntityExtractor
    — see tests/test_graph_construction.py), but any consumer's own
    implementation must also be usable: a minimal conforming stub should
    still type-check as both protocols via isinstance (they're
    @runtime_checkable)."""

    class StubGraphStore:
        def add_entities(self, entities): ...
        def add_relations(self, relations): ...
        def neighbors(self, entity_ids, *, hops=1, rel_types=None): return []
        def entities_for_chunks(self, chunk_ids): return {}
        def chunks_for_entities(self, entity_ids): return {}
        def entities_by_type(self, entity_type): return []

    class StubExtractor:
        def extract(self, chunks): return [], []

    assert isinstance(StubGraphStore(), GraphStore)
    assert isinstance(StubExtractor(), EntityExtractor)

    # the types themselves are constructible with the documented shape
    e = Entity(entity_id="e1", name="Attention", entity_type="concept", source_chunk_ids=("c1",))
    r = Relation(
        relation_id="r1",
        source_entity_id="e1",
        target_entity_id="e2",
        relation_type="relates_to",
        source_chunk_ids=("c1",),
    )
    gc = GraphContext(entities=("e1",), path=("e1", "r1", "e2"))
    assert e.entity_id == "e1"
    assert r.source_entity_id == "e1"
    assert gc.entities == ("e1",)
