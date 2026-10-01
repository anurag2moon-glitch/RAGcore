"""graph_search_modes(): wires GraphRetriever/ConceptRetriever/DriftRetriever
into a {"graph_local", "graph_global", "drift"} dict ready for
RoutedRetriever — integration test against real (zero-cost, in-memory)
SqliteGraphStore/SqliteConceptStore, mirroring test_graph_retrieval.py and
test_concept_retrieval.py's own approach exactly.
"""

from __future__ import annotations

from pathlib import Path

from ragcore.concepts.sqlite_store import SqliteConceptStore
from ragcore.concepts.types import Concept
from ragcore.corpus import Corpus
from ragcore.graph.sqlite_store import SqliteGraphStore
from ragcore.graph.types import Entity, Relation
from ragcore.retrieval.drift import DriftRetriever, graph_search_modes
from ragcore.retrieval.routing import RoutedRetriever
from ragcore.types import Filter, Retrieved
from tests.fakes import FakeEmbedder, InMemoryStore


class _FixedRetriever:
    def __init__(self, results: list[Retrieved]) -> None:
        self._results = results

    def retrieve(self, query: str, *, k: int = 10, filters: Filter | None = None) -> list[Retrieved]:
        return self._results[:k]


class _FakeRouter:
    def __init__(self, modes: list[str]) -> None:
        self._modes = modes

    def route(self, query, available_modes):
        return self._modes


def _corpus_with_chunks(md_book: Path) -> Corpus:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    return corpus


def test_graph_search_modes_returns_the_three_named_modes(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    base = _FixedRetriever([])
    graph = SqliteGraphStore(":memory:")
    concepts = SqliteConceptStore(":memory:")

    modes = graph_search_modes(
        base, corpus, graph_store=graph, concepts_store=concepts, embedder=corpus.embedder
    )

    assert set(modes) == {"graph_local", "graph_global", "drift"}
    assert isinstance(modes["drift"], DriftRetriever)


def test_graph_local_mode_expands_via_the_graph_store(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    all_chunks = corpus.store.all_chunks()
    seed_chunk, neighbor_chunk = all_chunks[0], all_chunks[1]

    graph = SqliteGraphStore(":memory:")
    graph.add_entities(
        [
            Entity(entity_id="e1", name="A", entity_type="person", source_chunk_ids=(seed_chunk.chunk_id,), extra={}),
            Entity(entity_id="e2", name="B", entity_type="person", source_chunk_ids=(neighbor_chunk.chunk_id,), extra={}),
        ]
    )
    graph.add_relations(
        [Relation(relation_id="r1", source_entity_id="e1", target_entity_id="e2", relation_type="knows", source_chunk_ids=(seed_chunk.chunk_id,), extra={})]
    )
    concepts = SqliteConceptStore(":memory:")
    base = _FixedRetriever([Retrieved(chunk=seed_chunk, score=0.9)])

    modes = graph_search_modes(
        base, corpus, graph_store=graph, concepts_store=concepts, embedder=corpus.embedder
    )
    results = modes["graph_local"].retrieve("q", k=5)

    assert {r.chunk.chunk_id for r in results} >= {seed_chunk.chunk_id, neighbor_chunk.chunk_id}


def test_graph_global_mode_surfaces_concept_linked_chunks(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    all_chunks = corpus.store.all_chunks()
    chunk = all_chunks[0]

    concepts = SqliteConceptStore(":memory:")
    concepts.add_chapter_concepts(
        source_id=chunk.meta.source_id,
        context_group_id=chunk.meta.context_group_id,
        fingerprint="fp",
        concepts=[
            Concept(
                concept_id="concept:test",
                label="a test concept",
                description="d",
                source_chunk_ids=(chunk.chunk_id,),
                vector=corpus.embedder.embed_query("a test concept"),
            )
        ],
        edges=[],
    )
    graph = SqliteGraphStore(":memory:")
    base = _FixedRetriever([Retrieved(chunk=chunk, score=0.9)])

    modes = graph_search_modes(
        base, corpus, graph_store=graph, concepts_store=concepts, embedder=corpus.embedder
    )
    results = modes["graph_global"].retrieve("a test concept", k=5)

    assert chunk.chunk_id in {r.chunk.chunk_id for r in results}


def test_drift_mode_fuses_local_and_global(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    all_chunks = corpus.store.all_chunks()
    seed_chunk, neighbor_chunk = all_chunks[0], all_chunks[1]

    graph = SqliteGraphStore(":memory:")
    graph.add_entities(
        [
            Entity(entity_id="e1", name="A", entity_type="person", source_chunk_ids=(seed_chunk.chunk_id,), extra={}),
            Entity(entity_id="e2", name="B", entity_type="person", source_chunk_ids=(neighbor_chunk.chunk_id,), extra={}),
        ]
    )
    graph.add_relations(
        [Relation(relation_id="r1", source_entity_id="e1", target_entity_id="e2", relation_type="knows", source_chunk_ids=(seed_chunk.chunk_id,), extra={})]
    )
    concepts = SqliteConceptStore(":memory:")
    base = _FixedRetriever([Retrieved(chunk=seed_chunk, score=0.9)])

    modes = graph_search_modes(
        base, corpus, graph_store=graph, concepts_store=concepts, embedder=corpus.embedder
    )
    results = modes["drift"].retrieve("q", k=5)

    assert seed_chunk.chunk_id in {r.chunk.chunk_id for r in results}


def test_composes_with_routed_retriever(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    all_chunks = corpus.store.all_chunks()
    chunk = all_chunks[0]

    graph = SqliteGraphStore(":memory:")
    concepts = SqliteConceptStore(":memory:")
    base = _FixedRetriever([Retrieved(chunk=chunk, score=0.9)])

    modes = graph_search_modes(
        base, corpus, graph_store=graph, concepts_store=concepts, embedder=corpus.embedder
    )
    modes["vector"] = base
    router = RoutedRetriever(modes, _FakeRouter(["graph_local"]))

    results = router.retrieve("q", k=5)

    assert router.last_route == ("graph_local",)
    assert chunk.chunk_id in {r.chunk.chunk_id for r in results}
