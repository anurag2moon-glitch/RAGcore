"""Concept layer at production scale: build_concepts -> detect_communities
-> build_community_summaries -> ConceptRetriever, against the shared,
already-ingested library (read-only — this module never mutates
shared_library, per the suite's rule that only fresh_corpus-based tests
mutate the index; see tests/integration/conftest.py).

A concept mention is planted using a real word from the shared library's
first book's first chunk as the marker (not an authoring-time offset), so
build_concepts has genuine chunk text to match against — the same
bridge-ground-truth-to-real-data principle
tests/integration/synth/library.py's find_chunks_with_marker uses for
entities, applied here without needing a parallel concept-ground-truth
system in the generator itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("networkx")

from ragcore.concepts.communities import build_community_summaries, detect_communities
from ragcore.concepts.pipeline import build_concepts
from ragcore.concepts.sqlite_store import SqliteConceptStore
from ragcore.concepts.types import CommunitySummary
from ragcore.retrieval.concept import ConceptRetriever
from ragcore.retrieval.vector import VectorRetriever
from tests.integration.conftest import SharedLibrary
from tests.integration.doubles import ConceptMention, MockConceptExtractor


def _build_concept_script(shared_library: SharedLibrary) -> dict[str, list[ConceptMention]]:
    key = shared_library.book_keys[0]
    spec = shared_library.book(key)
    source = next(
        s for s in shared_library.corpus.sources() if Path(s.path).resolve() == spec.path.resolve()
    )
    chunks = shared_library.corpus.store.get_by_source(source.source_id)
    assert chunks, "test premise broken: the shared library's first book has no chunks"

    first_chunk = chunks[0]
    marker = first_chunk.text.split()[0]
    return {
        first_chunk.meta.context_group_id: [
            ConceptMention(
                canonical_id="shared-theme",
                label="a recurring narrative theme across chapters",
                description="Planted for the integration suite's concept-layer test.",
                marker=marker,
            )
        ]
    }


class _FakeSummarizer:
    model = "fake-summarizer"

    def summarize(self, *, community_id, concepts, chunks) -> CommunitySummary:
        return CommunitySummary(
            community_id=community_id,
            label="L",
            summary="S",
            agreements=(),
            divergences=(),
            concept_ids=tuple(c.concept_id for c in concepts),
            source_chunk_ids=tuple(c.chunk_id for c in chunks),
            source_ids=tuple(sorted({c.meta.source_id for c in chunks})),
            model=self.model,
            prompt_version="v1",
            created_at="now",
        )


def test_build_concepts_and_communities_run_at_scale_without_error(shared_library: SharedLibrary) -> None:
    script = _build_concept_script(shared_library)
    extractor = MockConceptExtractor(script)
    store = SqliteConceptStore(":memory:")

    report = build_concepts(shared_library.corpus, store, extractor)

    assert report.chapters_processed > 0
    assert report.errors == ()
    # every chapter of every ingested book was visited exactly once on a
    # fresh store (nothing skipped yet)
    assert report.chapters_skipped == 0
    assert len(extractor.calls) == report.chapters_processed

    community_report = detect_communities(store, seed=0)
    assert community_report.ran is True
    assert community_report.concept_count == len(store.all_concepts())

    n_summaries = build_community_summaries(shared_library.corpus, store, _FakeSummarizer())
    assert n_summaries == community_report.community_count


def test_build_concepts_is_idempotent_on_rerun(shared_library: SharedLibrary) -> None:
    script = _build_concept_script(shared_library)
    store = SqliteConceptStore(":memory:")
    extractor1 = MockConceptExtractor(script)
    report1 = build_concepts(shared_library.corpus, store, extractor1)

    extractor2 = MockConceptExtractor(script)
    report2 = build_concepts(shared_library.corpus, store, extractor2)

    assert report2.chapters_processed == 0
    assert report2.chapters_skipped == report1.chapters_processed
    assert len(extractor2.calls) == 0  # every chapter's checkpoint already matched


def test_concept_retriever_composes_with_a_real_vector_retriever_at_scale(
    shared_library: SharedLibrary,
) -> None:
    script = _build_concept_script(shared_library)
    store = SqliteConceptStore(":memory:")
    build_concepts(shared_library.corpus, store, MockConceptExtractor(script))
    detect_communities(store, seed=0)

    inner = VectorRetriever(shared_library.corpus.embedder, shared_library.corpus.store)
    retriever = ConceptRetriever(inner, store, shared_library.corpus, shared_library.corpus.embedder)

    results = retriever.retrieve("a recurring narrative theme across chapters", k=10)
    assert results
    assert len(results) <= 10
