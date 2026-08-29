"""GraphRetriever composing with a fake vector retriever and a real (but
zero-cost) SqliteGraphStore(":memory:"). No network, no embeddings needed
here at all — the inner retriever is a fixed stub, so this tests exactly
the expansion/merge/scoring logic in ragcore/retrieval/graph.py.
"""

from __future__ import annotations

from pathlib import Path

from ragcore.corpus import Corpus
from ragcore.graph.sqlite_store import SqliteGraphStore
from ragcore.graph.types import Entity, Relation
from ragcore.retrieval.graph import GraphRetriever
from ragcore.types import Filter, Retrieved
from tests.fakes import FakeEmbedder, InMemoryStore


class _FixedRetriever:
    """Retriever stub returning a pre-set list of Retrieved for any query."""

    def __init__(self, results: list[Retrieved]) -> None:
        self._results = results
        self.calls: list[tuple[str, int, Filter | None]] = []

    def retrieve(self, query: str, *, k: int = 10, filters: Filter | None = None) -> list[Retrieved]:
        self.calls.append((query, k, filters))
        return self._results[:k]


def _corpus_with_chunks(md_book: Path) -> Corpus:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    return corpus


def test_graph_retriever_returns_seeds_unchanged_when_graph_has_nothing(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    all_chunks = corpus.store.all_chunks()
    seed = Retrieved(chunk=all_chunks[0], score=0.9)

    inner = _FixedRetriever([seed])
    graph = SqliteGraphStore(":memory:")  # empty — no entities at all
    retriever = GraphRetriever(inner, graph, corpus)

    results = retriever.retrieve("anything", k=5)

    assert results == [seed]
    assert results[0].graph is None


def test_graph_retriever_expands_one_hop_and_scores_below_seeds(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    all_chunks = corpus.store.all_chunks()
    seed_chunk, neighbor_chunk = all_chunks[0], all_chunks[1]
    assert seed_chunk.chunk_id != neighbor_chunk.chunk_id

    graph = SqliteGraphStore(":memory:")
    graph.add_entities(
        [
            Entity(entity_id="entity:person:a", name="A", entity_type="person",
                   source_chunk_ids=(seed_chunk.chunk_id,)),
            Entity(entity_id="entity:person:b", name="B", entity_type="person",
                   source_chunk_ids=(neighbor_chunk.chunk_id,)),
        ]
    )
    graph.add_relations(
        [
            Relation(relation_id="relation:friend_of:entity:person:a:entity:person:b",
                      source_entity_id="entity:person:a", target_entity_id="entity:person:b",
                      relation_type="friend_of", source_chunk_ids=(seed_chunk.chunk_id,))
        ]
    )

    seed = Retrieved(chunk=seed_chunk, score=0.9)
    inner = _FixedRetriever([seed])
    retriever = GraphRetriever(inner, graph, corpus, expansion_score_decay=0.5)

    results = retriever.retrieve("anything", k=5)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == seed_chunk.chunk_id
    assert results[0].graph is None  # seeds don't get graph provenance
    assert results[1].chunk.chunk_id == neighbor_chunk.chunk_id
    assert results[1].score == 0.9 * 0.5
    assert results[1].graph is not None
    assert results[1].graph.entities == ("entity:person:b",)


def test_graph_retriever_respects_k_and_expansion_budget(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    all_chunks = corpus.store.all_chunks()
    seed_chunk = all_chunks[0]
    neighbor_chunks = all_chunks[1:4]  # three potential expansions

    graph = SqliteGraphStore(":memory:")
    graph.add_entities(
        [Entity(entity_id="entity:person:seed", name="Seed", entity_type="person",
                source_chunk_ids=(seed_chunk.chunk_id,))]
        + [
            Entity(entity_id=f"entity:person:n{i}", name=f"N{i}", entity_type="person",
                   source_chunk_ids=(c.chunk_id,))
            for i, c in enumerate(neighbor_chunks)
        ]
    )
    graph.add_relations(
        [
            Relation(relation_id=f"relation:rel:entity:person:seed:entity:person:n{i}",
                      source_entity_id="entity:person:seed", target_entity_id=f"entity:person:n{i}",
                      relation_type="rel", source_chunk_ids=(seed_chunk.chunk_id,))
            for i in range(len(neighbor_chunks))
        ]
    )

    seed = Retrieved(chunk=seed_chunk, score=1.0)
    inner = _FixedRetriever([seed])

    # budget=1 caps expansion to one extra chunk even though 3 are graph-reachable
    retriever = GraphRetriever(inner, graph, corpus, expansion_budget=1)
    results = retriever.retrieve("anything", k=10)
    assert len(results) == 2  # 1 seed + 1 expansion (budget-capped)

    # k=1 truncates to just the top result (the seed, since expansions score lower)
    results_k1 = retriever.retrieve("anything", k=1)
    assert len(results_k1) == 1
    assert results_k1[0].chunk.chunk_id == seed_chunk.chunk_id
