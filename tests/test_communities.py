"""detect_communities: deterministic clustering with a fixed seed, stable
community ids under unchanged membership, and the corpus-global checkpoint
that skips reclustering when the concept graph hasn't changed.
build_community_summaries: per-community checkpointing and the embedding
attached to a summary at build time (see ragcore.concepts.communities'
module docstring for why summarization is checkpointed independently of
detection).
"""

from __future__ import annotations

import pytest

pytest.importorskip("networkx")

from ragcore.concepts.communities import build_community_summaries, detect_communities
from ragcore.concepts.sqlite_store import SqliteConceptStore
from ragcore.concepts.types import Concept, ConceptEdge, CommunitySummary
from tests.fakes import FakeEmbedder


def _seed_two_disconnected_pairs(store: SqliteConceptStore) -> None:
    concepts = [
        Concept(concept_id=f"concept:c{i}", label=f"idea {i}", description="d", source_chunk_ids=("c1",))
        for i in range(4)
    ]
    edges = [
        ConceptEdge(edge_id="edge:concept:c0:concept:c1", source_concept_id="concept:c0",
                    target_concept_id="concept:c1", weight=0, source_chunk_ids=("c1",)),
        ConceptEdge(edge_id="edge:concept:c2:concept:c3", source_concept_id="concept:c2",
                    target_concept_id="concept:c3", weight=0, source_chunk_ids=("c1",)),
    ]
    store.add_chapter_concepts(source_id="src", context_group_id="g", fingerprint="fp", concepts=concepts, edges=edges)


def test_detect_communities_groups_connected_concepts_together() -> None:
    store = SqliteConceptStore(":memory:")
    _seed_two_disconnected_pairs(store)

    report = detect_communities(store, seed=0)

    assert report.ran is True
    assert report.community_count == 2
    c0_community = store.community_id_for_concept("concept:c0")
    c1_community = store.community_id_for_concept("concept:c1")
    c2_community = store.community_id_for_concept("concept:c2")
    assert c0_community == c1_community
    assert c0_community != c2_community


def test_community_id_is_stable_when_membership_is_unchanged() -> None:
    store = SqliteConceptStore(":memory:")
    _seed_two_disconnected_pairs(store)
    detect_communities(store, seed=0)
    first_id = store.community_id_for_concept("concept:c0")

    # Re-run with the exact same graph: same seed, same edges -> same id.
    detect_communities(store, seed=0, force=True)
    second_id = store.community_id_for_concept("concept:c0")

    assert first_id == second_id


def test_detect_communities_skips_when_the_graph_fingerprint_is_unchanged() -> None:
    store = SqliteConceptStore(":memory:")
    _seed_two_disconnected_pairs(store)

    report1 = detect_communities(store, seed=0)
    assert report1.ran is True

    report2 = detect_communities(store, seed=0)
    assert report2.ran is False
    assert report2.community_count == report1.community_count


def test_detect_communities_reclusters_when_a_new_concept_is_added() -> None:
    store = SqliteConceptStore(":memory:")
    _seed_two_disconnected_pairs(store)
    detect_communities(store, seed=0)

    new_concept = Concept(concept_id="concept:c4", label="idea 4", description="d", source_chunk_ids=("c2",))
    store.add_chapter_concepts(source_id="src", context_group_id="g2", fingerprint="fp2", concepts=[new_concept], edges=[])

    report = detect_communities(store, seed=0)
    assert report.ran is True
    assert report.concept_count == 5


class _FakeSummarizer:
    model = "fake-summarizer"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def summarize(self, *, community_id, concepts, chunks) -> CommunitySummary:
        self.calls.append(community_id)
        return CommunitySummary(
            community_id=community_id, label="L", summary="S", agreements=(), divergences=(),
            concept_ids=tuple(c.concept_id for c in concepts),
            source_chunk_ids=tuple(c.chunk_id for c in chunks),
            source_ids=tuple(sorted({c.meta.source_id for c in chunks})),
            model=self.model, prompt_version="v1", created_at="now",
        )


def _corpus_with_one_chunk(md_book):
    from ragcore.corpus import Corpus
    from tests.fakes import InMemoryStore

    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    return corpus


def test_build_community_summaries_computes_once_and_embeds_the_summary(md_book) -> None:
    corpus = _corpus_with_one_chunk(md_book)
    chunk_id = corpus.store.all_chunks()[0].chunk_id

    store = SqliteConceptStore(":memory:")
    concept = Concept(concept_id="concept:c0", label="idea", description="d", source_chunk_ids=(chunk_id,))
    store.add_chapter_concepts(source_id="src", context_group_id="g", fingerprint="fp", concepts=[concept], edges=[])
    detect_communities(store, seed=0)

    summarizer = _FakeSummarizer()
    n = build_community_summaries(corpus, store, summarizer)
    assert n == 1
    assert len(summarizer.calls) == 1

    community_id = store.all_community_ids()[0]
    summary = store.community_summary(community_id)
    assert summary is not None
    assert summary.vector is not None  # embedded at build time, not left None

    # Re-run without force: already has a summary, must not recompute.
    n2 = build_community_summaries(corpus, store, summarizer)
    assert n2 == 0
    assert len(summarizer.calls) == 1


def test_build_community_summaries_force_recomputes_existing_summaries(md_book) -> None:
    corpus = _corpus_with_one_chunk(md_book)
    chunk_id = corpus.store.all_chunks()[0].chunk_id

    store = SqliteConceptStore(":memory:")
    concept = Concept(concept_id="concept:c0", label="idea", description="d", source_chunk_ids=(chunk_id,))
    store.add_chapter_concepts(source_id="src", context_group_id="g", fingerprint="fp", concepts=[concept], edges=[])
    detect_communities(store, seed=0)

    summarizer = _FakeSummarizer()
    build_community_summaries(corpus, store, summarizer)
    n2 = build_community_summaries(corpus, store, summarizer, force=True)
    assert n2 == 1
    assert len(summarizer.calls) == 2
