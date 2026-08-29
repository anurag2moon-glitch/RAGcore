"""build_graph() and EmbeddingEntityResolver, against real (but zero-cost)
implementations: SqliteGraphStore (stdlib sqlite3, ":memory:") and
FakeEmbedder — no network, no anthropic/voyageai. FakeEntityExtractor
stands in for the real ClaudeEntityExtractor so orchestration (chapter
grouping, merge-vs-replace, id remapping after a resolver merge) is tested
independently of what a real LLM would actually extract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ragcore.corpus import Corpus
from ragcore.graph.pipeline import build_graph
from ragcore.graph.resolver import EmbeddingEntityResolver
from ragcore.graph.sqlite_store import SqliteGraphStore
from ragcore.graph.types import Entity, Relation
from ragcore.types import Chunk
from tests.fakes import FakeEmbedder, FakeEntityExtractor, InMemoryStore


def _build_corpus(dim: int = 8) -> tuple[Corpus, InMemoryStore]:
    store = InMemoryStore(dim=dim)
    corpus = Corpus(store, FakeEmbedder(dim=dim))
    return corpus, store


def test_build_graph_groups_by_chapter_and_records_provenance(md_book: Path) -> None:
    corpus, _ = _build_corpus()
    corpus.ingest(md_book)

    def fn(chunks: Sequence[Chunk]) -> tuple[list[Entity], list[Relation]]:
        chunk_ids = tuple(c.chunk_id for c in chunks)
        entity = Entity(
            entity_id="entity:concept:attention",
            name="Attention",
            entity_type="concept",
            source_chunk_ids=chunk_ids,
        )
        return [entity], []

    extractor = FakeEntityExtractor(fn)
    graph = SqliteGraphStore(":memory:")

    report = build_graph(corpus, graph, extractor)

    # one extract() call per chapter (ChunkGroup), not one per chunk and not
    # one for the whole book
    assert len(extractor.calls) >= 2
    assert report.chunks_processed == sum(len(c) for c in extractor.calls)
    assert report.entities_extracted == len(extractor.calls)  # one entity per call
    assert report.relations_extracted == 0
    assert report.entities_merged == 0

    # provenance: the entity's chunk_ids round-trip to real chapter metadata
    entities_by_chunk = graph.entities_for_chunks(extractor.calls[0])
    assert extractor.calls[0][0] in entities_by_chunk
    found = entities_by_chunk[extractor.calls[0][0]]
    assert any(e.entity_id == "entity:concept:attention" for e in found)

    chunks_for_entity = graph.chunks_for_entities(["entity:concept:attention"])
    hydrated = corpus.get(chunks_for_entity["entity:concept:attention"])
    assert hydrated
    assert all(c.meta.chapter_index is not None for c in hydrated)


def test_build_graph_merges_repeated_entity_across_chapters(md_book: Path) -> None:
    """Same content-addressed entity_id extracted from two different
    chapters must accumulate provenance (merge), not overwrite it."""
    corpus, _ = _build_corpus()
    corpus.ingest(md_book)

    def fn(chunks: Sequence[Chunk]) -> tuple[list[Entity], list[Relation]]:
        entity = Entity(
            entity_id="entity:concept:attention",
            name="Attention",
            entity_type="concept",
            source_chunk_ids=(chunks[0].chunk_id,),
        )
        return [entity], []

    extractor = FakeEntityExtractor(fn)
    graph = SqliteGraphStore(":memory:")
    build_graph(corpus, graph, extractor)

    assert len(extractor.calls) >= 2  # at least two chapters -> two extract() calls
    all_chunk_ids = graph.chunks_for_entities(["entity:concept:attention"])["entity:concept:attention"]
    # every chapter's first chunk_id contributed, none clobbered the last
    expected = {calls[0] for calls in extractor.calls}
    assert set(all_chunk_ids) == expected


def test_build_graph_remaps_relation_ids_after_a_resolver_merge(md_book: Path) -> None:
    """When the resolver merges a new candidate into an existing entity,
    any relation extracted in that same call (which references the
    candidate's pre-merge id) must be remapped to the merged id too."""
    corpus, _ = _build_corpus()
    corpus.ingest(md_book)
    graph = SqliteGraphStore(":memory:")

    # Seed an existing "person:mr-darcy" entity directly, as if a prior run
    # had already extracted it under this exact canonical name.
    graph.add_entities(
        [
            Entity(
                entity_id="entity:person:mr-darcy",
                name="Mr. Darcy",
                entity_type="person",
                source_chunk_ids=("seed-chunk",),
            )
        ]
    )

    calls = {"n": 0}

    def fn(chunks: Sequence[Chunk]) -> tuple[list[Entity], list[Relation]]:
        calls["n"] += 1
        if calls["n"] > 1:
            return [], []
        # A differently-worded candidate for the same person, plus a
        # relation that (per the extractor's own contract) references the
        # candidate's own pre-resolution entity_id.
        candidate = Entity(
            entity_id="entity:person:fitzwilliam-darcy",
            name="Fitzwilliam Darcy",
            entity_type="person",
            source_chunk_ids=(chunks[0].chunk_id,),
        )
        bingley = Entity(
            entity_id="entity:person:bingley",
            name="Bingley",
            entity_type="person",
            source_chunk_ids=(chunks[0].chunk_id,),
        )
        relation = Relation(
            relation_id=f"relation:friend_of:{candidate.entity_id}:{bingley.entity_id}",
            source_entity_id=candidate.entity_id,
            target_entity_id=bingley.entity_id,
            relation_type="friend_of",
            source_chunk_ids=(chunks[0].chunk_id,),
        )
        return [candidate, bingley], [relation]

    extractor = FakeEntityExtractor(fn)

    class _AlwaysMergeResolver:
        """Forces the "Fitzwilliam Darcy" candidate to merge into the
        pre-seeded "Mr. Darcy" entity, deterministically (no embedding call
        needed for this test — it's testing the remap plumbing in
        build_graph, not resolution quality itself)."""

        def resolve(self, candidate, existing):
            if candidate.entity_type == "person" and candidate.name == "Fitzwilliam Darcy":
                return "entity:person:mr-darcy"
            return None

    report = build_graph(corpus, graph, extractor, resolver=_AlwaysMergeResolver())

    assert report.entities_merged == 1
    # the relation must now point at the merged id, not the pre-merge one
    relations = graph.neighbors(["entity:person:mr-darcy"])
    assert len(relations) == 1
    assert relations[0].source_entity_id == "entity:person:mr-darcy"
    assert relations[0].target_entity_id == "entity:person:bingley"
    assert relations[0].relation_id == "relation:friend_of:entity:person:mr-darcy:entity:person:bingley"

    # the pre-merge id was never persisted as its own entity
    assert graph.entities_for_chunks(["seed-chunk"])["seed-chunk"][0].entity_id == "entity:person:mr-darcy"


def test_embedding_entity_resolver_merges_similar_rejects_dissimilar() -> None:
    """Direct unit test of EmbeddingEntityResolver, with a controllable stub
    embedder (not the hash-based FakeEmbedder, whose vectors are
    uncorrelated with meaning by construction) so cosine similarity is
    exactly what the test sets it to be."""

    class _StubEmbedder:
        name = "stub"
        dim = 2

        def __init__(self, vectors: dict[str, list[float]]) -> None:
            self._vectors = vectors

        def embed_groups(self, groups, *, input_type="document"):
            return [[self._vectors[text] for text in group] for group in groups]

        def embed_query(self, text):
            return self._vectors[text]

    darcy = Entity(entity_id="entity:person:mr-darcy", name="Mr. Darcy", entity_type="person",
                    source_chunk_ids=("c1",))
    bingley = Entity(entity_id="entity:person:bingley", name="Bingley", entity_type="person",
                      source_chunk_ids=("c2",))

    similar_candidate = Entity(entity_id="entity:person:fitzwilliam-darcy", name="Fitzwilliam Darcy",
                                entity_type="person", source_chunk_ids=("c3",))
    dissimilar_candidate = Entity(entity_id="entity:person:someone-else", name="Someone Else",
                                   entity_type="person", source_chunk_ids=("c4",))

    vectors = {
        "person: Mr. Darcy": [1.0, 0.0],
        "person: Bingley": [0.0, 1.0],
        "person: Fitzwilliam Darcy": [0.99, 0.01],   # nearly identical to Mr. Darcy
        "person: Someone Else": [-1.0, 0.0],          # opposite of everything
    }
    resolver = EmbeddingEntityResolver(_StubEmbedder(vectors), threshold=0.87)

    existing = [darcy, bingley]
    assert resolver.resolve(similar_candidate, existing) == "entity:person:mr-darcy"
    assert resolver.resolve(dissimilar_candidate, existing) is None
