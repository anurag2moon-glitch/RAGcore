"""SqliteConceptStore: merge-not-replace on re-extraction, provenance
accumulation, and edge weight derived from distinct co-occurring chunks
(never an incrementally-updated counter — see the module docstring in
ragcore/concepts/sqlite_store.py for why an incremented counter would break
idempotency on a re-run of the same chapter).
"""

from __future__ import annotations

from ragcore.concepts.sqlite_store import SqliteConceptStore
from ragcore.concepts.types import Concept, ConceptEdge


def test_add_chapter_concepts_merges_source_chunk_ids_across_calls() -> None:
    store = SqliteConceptStore(":memory:")
    concept = Concept(
        concept_id="concept:ambition", label="ambition", description="d", source_chunk_ids=("c1",)
    )
    store.add_chapter_concepts(
        source_id="src", context_group_id="g1", fingerprint="fp1", concepts=[concept], edges=[]
    )
    store.add_chapter_concepts(
        source_id="src",
        context_group_id="g2",
        fingerprint="fp2",
        concepts=[
            Concept(concept_id="concept:ambition", label="ambition", description="d", source_chunk_ids=("c2",))
        ],
        edges=[],
    )

    stored = store.concept("concept:ambition")
    assert set(stored.source_chunk_ids) == {"c1", "c2"}


def test_add_chapter_concepts_never_clobbers_a_stored_vector_with_none() -> None:
    store = SqliteConceptStore(":memory:")
    with_vector = Concept(
        concept_id="concept:ambition", label="ambition", description="d",
        source_chunk_ids=("c1",), vector=(1.0, 0.0),
    )
    store.add_chapter_concepts(
        source_id="src", context_group_id="g1", fingerprint="fp1", concepts=[with_vector], edges=[]
    )
    # A merge candidate (e.g. an exact re-mention in another chapter) never
    # carries its own vector — see Concept.vector's docstring.
    without_vector = Concept(
        concept_id="concept:ambition", label="ambition", description="d", source_chunk_ids=("c2",)
    )
    store.add_chapter_concepts(
        source_id="src", context_group_id="g2", fingerprint="fp2", concepts=[without_vector], edges=[]
    )

    assert store.concept_vector("concept:ambition") == (1.0, 0.0)


def test_edge_weight_is_derived_from_distinct_chunks_not_an_incremented_counter() -> None:
    """Re-running the exact same chapter (e.g. after a checkpoint reset)
    must not double the edge's weight — the junction table's INSERT OR
    IGNORE makes chunk association idempotent, and weight is recomputed as
    COUNT(DISTINCT chunk_id) from it, not accumulated."""
    store = SqliteConceptStore(":memory:")
    a = Concept(concept_id="concept:a", label="a", description="d", source_chunk_ids=("c1",))
    b = Concept(concept_id="concept:b", label="b", description="d", source_chunk_ids=("c1",))
    edge = ConceptEdge(
        edge_id="edge:concept:a:concept:b", source_concept_id="concept:a",
        target_concept_id="concept:b", weight=0, source_chunk_ids=("c1",),
    )

    store.add_chapter_concepts(
        source_id="src", context_group_id="g", fingerprint="fp", concepts=[a, b], edges=[edge]
    )
    store.add_chapter_concepts(  # same chapter, re-processed
        source_id="src", context_group_id="g", fingerprint="fp", concepts=[a, b], edges=[edge]
    )

    stored_edge = next(e for e in store.all_edges() if e.edge_id == "edge:concept:a:concept:b")
    assert stored_edge.weight == 1


def test_edge_weight_counts_every_distinct_contributing_chunk() -> None:
    store = SqliteConceptStore(":memory:")
    a = Concept(concept_id="concept:a", label="a", description="d", source_chunk_ids=("c1", "c2"))
    b = Concept(concept_id="concept:b", label="b", description="d", source_chunk_ids=("c1", "c2"))
    edge = ConceptEdge(
        edge_id="edge:concept:a:concept:b", source_concept_id="concept:a",
        target_concept_id="concept:b", weight=0, source_chunk_ids=("c1", "c2"),
    )
    store.add_chapter_concepts(
        source_id="src", context_group_id="g", fingerprint="fp", concepts=[a, b], edges=[edge]
    )

    stored_edge = store.all_edges()[0]
    assert stored_edge.weight == 2


def test_checkpoint_round_trips_and_mark_chapter_failed_records_status() -> None:
    store = SqliteConceptStore(":memory:")
    assert store.checkpoint("src", "g1") is None

    store.mark_chapter_failed(source_id="src", context_group_id="g1", fingerprint="fp1", error="boom")
    assert store.checkpoint("src", "g1") is None  # a failed attempt is never a checkpoint

    store.add_chapter_concepts(source_id="src", context_group_id="g1", fingerprint="fp1", concepts=[], edges=[])
    assert store.checkpoint("src", "g1") == "fp1"


def test_communities_for_concepts_returns_distinct_summaries() -> None:
    from ragcore.concepts.types import CommunitySummary

    store = SqliteConceptStore(":memory:")
    store.set_communities(
        run_fingerprint="fp", assignments={"concept:a": "community:x", "concept:b": "community:x"}
    )
    store.put_community_summary(
        CommunitySummary(
            community_id="community:x", label="L", summary="S", agreements=(), divergences=(),
            concept_ids=("concept:a", "concept:b"), source_chunk_ids=("c1",), source_ids=("src",),
            model="m", prompt_version="v1", created_at="now",
        )
    )

    summaries = store.communities_for_concepts(["concept:a", "concept:b"])
    assert len(summaries) == 1
    assert summaries[0].community_id == "community:x"
