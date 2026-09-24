"""SqliteConceptStore.remove_chunk_ids/remove_source: pruning provenance
and deleting orphaned concepts/edges/community assignments, plus
build_community_summaries' chunk-fingerprint staleness fix and
detect_communities' orphan-summary pruning — all new capabilities item 7
(incremental ingestion) needs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("networkx")

from ragcore.concepts.communities import build_community_summaries, detect_communities
from ragcore.concepts.sqlite_store import SqliteConceptStore
from ragcore.concepts.types import Concept, ConceptEdge, CommunitySummary
from ragcore.corpus import Corpus
from tests.fakes import FakeEmbedder, InMemoryStore


def test_remove_source_prunes_provenance_but_keeps_concepts_with_other_sources() -> None:
    store = SqliteConceptStore(":memory:")
    store.add_chapter_concepts(
        source_id="src1", context_group_id="g1", fingerprint="fp1",
        concepts=[Concept(concept_id="concept:a", label="a", description="d", source_chunk_ids=("src1:0",))],
        edges=[],
    )
    store.add_chapter_concepts(
        source_id="src2", context_group_id="g2", fingerprint="fp2",
        concepts=[Concept(concept_id="concept:a", label="a", description="d", source_chunk_ids=("src2:0",))],
        edges=[],
    )

    store.remove_source("src1")

    remaining = store.concept("concept:a")
    assert remaining is not None
    assert remaining.source_chunk_ids == ("src2:0",)


def test_remove_source_deletes_a_concept_left_with_zero_chunks() -> None:
    store = SqliteConceptStore(":memory:")
    store.add_chapter_concepts(
        source_id="src1", context_group_id="g1", fingerprint="fp1",
        concepts=[Concept(concept_id="concept:a", label="a", description="d", source_chunk_ids=("src1:0",))],
        edges=[],
    )

    store.remove_source("src1")

    assert store.concept("concept:a") is None
    assert store.checkpoint("src1", "g1") is None


def test_remove_source_deletes_an_edge_when_an_endpoint_concept_is_deleted() -> None:
    store = SqliteConceptStore(":memory:")
    a = Concept(concept_id="concept:a", label="a", description="d", source_chunk_ids=("src1:0",))
    b = Concept(concept_id="concept:b", label="b", description="d", source_chunk_ids=("src1:0", "src2:0"))
    edge = ConceptEdge(edge_id="edge:concept:a:concept:b", source_concept_id="concept:a",
                        target_concept_id="concept:b", weight=0, source_chunk_ids=("src1:0",))
    store.add_chapter_concepts(source_id="src1", context_group_id="g1", fingerprint="fp1",
                                concepts=[a, b], edges=[edge])

    store.remove_source("src1")

    assert store.concept("concept:a") is None  # fully orphaned, deleted
    assert store.concept("concept:b") is not None  # still has src2:0
    assert store.all_edges() == []  # edge's endpoint concept:a no longer exists


def test_remove_source_also_deletes_stale_community_assignment() -> None:
    store = SqliteConceptStore(":memory:")
    store.add_chapter_concepts(
        source_id="src1", context_group_id="g1", fingerprint="fp1",
        concepts=[Concept(concept_id="concept:a", label="a", description="d", source_chunk_ids=("src1:0",))],
        edges=[],
    )
    store.set_communities(run_fingerprint="cfp", assignments={"concept:a": "community:x"})

    store.remove_source("src1")

    assert store.community_id_for_concept("concept:a") is None


class _FakeSummarizer:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def summarize(self, *, community_id, concepts, chunks) -> CommunitySummary:
        self.calls += 1
        return CommunitySummary(
            community_id=community_id, label="L", summary=f"summary v{self.calls}",
            agreements=(), divergences=(), concept_ids=tuple(c.concept_id for c in concepts),
            source_chunk_ids=tuple(c.chunk_id for c in chunks),
            source_ids=tuple(sorted({c.meta.source_id for c in chunks})),
            model=self.model, prompt_version="v1", created_at="now",
        )


def _corpus_with_chunk(md_book: Path):
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    return corpus


def test_build_community_summaries_recomputes_when_reachable_chunks_change(md_book: Path) -> None:
    corpus = _corpus_with_chunk(md_book)
    all_chunks = corpus.store.all_chunks()
    chunk_a, chunk_b = all_chunks[0], all_chunks[1]

    store = SqliteConceptStore(":memory:")
    concept = Concept(concept_id="concept:a", label="a", description="d", source_chunk_ids=(chunk_a.chunk_id,))
    store.add_chapter_concepts(source_id="src", context_group_id="g", fingerprint="fp", concepts=[concept], edges=[])
    detect_communities(store, seed=0)

    summarizer = _FakeSummarizer()
    n1 = build_community_summaries(corpus, store, summarizer)
    assert n1 == 1
    assert summarizer.calls == 1

    # Re-run with nothing changed: must skip (fingerprint matches).
    n2 = build_community_summaries(corpus, store, summarizer)
    assert n2 == 0
    assert summarizer.calls == 1

    # Now the concept's reachable chunk set grows (e.g. a re-run found it in
    # another chapter too) — the community's chunk_fingerprint changes even
    # though its membership (concept:a) did not, so it must recompute.
    concept2 = Concept(concept_id="concept:a", label="a", description="d", source_chunk_ids=(chunk_b.chunk_id,))
    store.add_chapter_concepts(source_id="src", context_group_id="g2", fingerprint="fp2", concepts=[concept2], edges=[])

    n3 = build_community_summaries(corpus, store, summarizer)
    assert n3 == 1
    assert summarizer.calls == 2


def test_detect_communities_prunes_orphaned_summaries_after_a_real_recluster(md_book: Path) -> None:
    corpus = _corpus_with_chunk(md_book)
    chunk = corpus.store.all_chunks()[0]
    real_source_id = chunk.meta.source_id  # must match the chunk_id's own prefix for remove_source to find it
    store = SqliteConceptStore(":memory:")
    concept = Concept(concept_id="concept:a", label="a", description="d", source_chunk_ids=(chunk.chunk_id,))
    store.add_chapter_concepts(source_id=real_source_id, context_group_id="g", fingerprint="fp", concepts=[concept], edges=[])

    detect_communities(store, seed=0)
    build_community_summaries(corpus, store, _FakeSummarizer())
    old_community_id = store.all_community_ids()[0]
    assert store.community_summary(old_community_id) is not None

    # Removing the source deletes the concept, which changes the concept
    # graph entirely — force a recluster (empty graph now).
    store.remove_source(real_source_id)
    detect_communities(store, seed=0, force=True)

    assert store.community_summary(old_community_id) is None
