"""ConceptRetriever composing with a fake vector retriever and a real
(zero-cost) SqliteConceptStore(":memory:") — mirrors
test_graph_retrieval.py's approach for GraphRetriever exactly. Verifies the
k contract, that a query with no concept signal still returns a full k, that
concept-linked chunks get a real cosine-derived score on the same scale as
vector hits, and that community summaries arrive as synthetic chunks.
"""

from __future__ import annotations

from pathlib import Path

from ragcore.corpus import Corpus
from ragcore.concepts.sqlite_store import SqliteConceptStore
from ragcore.concepts.types import Concept, CommunitySummary
from ragcore.retrieval.concept import ConceptRetriever
from ragcore.types import Filter, Retrieved
from tests.fakes import FakeEmbedder, InMemoryStore


class _FixedRetriever:
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


def test_no_concept_signal_still_returns_a_full_k(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    all_chunks = corpus.store.all_chunks()
    seeds = [Retrieved(chunk=c, score=1.0 - i * 0.1) for i, c in enumerate(all_chunks)]

    inner = _FixedRetriever(seeds)
    store = SqliteConceptStore(":memory:")  # empty — no concepts at all
    retriever = ConceptRetriever(inner, store, corpus, FakeEmbedder(dim=8))

    results = retriever.retrieve("anything", k=len(seeds))

    assert len(results) == len(seeds)
    assert all(r.chunk.meta.extra.get("ragcore_kind") is None for r in results)


def test_concept_linked_chunk_is_pulled_in_with_a_real_cosine_score(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    all_chunks = corpus.store.all_chunks()
    seed_chunk, linked_chunk = all_chunks[0], all_chunks[1]
    embedder = FakeEmbedder(dim=8)
    query = "some query text"

    store = SqliteConceptStore(":memory:")
    concept = Concept(
        concept_id="concept:a", label="idea", description="d",
        source_chunk_ids=(linked_chunk.chunk_id,), vector=tuple(embedder.embed_query(query)),
    )
    store.add_chapter_concepts(
        source_id=seed_chunk.meta.source_id, context_group_id="g", fingerprint="fp",
        concepts=[concept], edges=[],
    )

    seed = Retrieved(chunk=seed_chunk, score=0.9)
    inner = _FixedRetriever([seed])
    retriever = ConceptRetriever(inner, store, corpus, embedder)

    results = retriever.retrieve(query, k=5)

    linked = next((r for r in results if r.chunk.chunk_id == linked_chunk.chunk_id), None)
    assert linked is not None
    assert 0.0 < linked.score <= 1.0  # a real 1/(1+distance) score, not an invented constant
    assert linked.chunk.meta.extra.get("ragcore_kind") is None  # a real passage, not synthetic


def test_relevant_community_summary_arrives_as_a_synthetic_chunk(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    all_chunks = corpus.store.all_chunks()
    seed_chunk = all_chunks[0]
    embedder = FakeEmbedder(dim=8)
    query = "summary query"

    store = SqliteConceptStore(":memory:")
    concept = Concept(
        concept_id="concept:a", label="idea", description="d",
        source_chunk_ids=(seed_chunk.chunk_id,), vector=tuple(embedder.embed_query(query)),
    )
    store.add_chapter_concepts(
        source_id=seed_chunk.meta.source_id, context_group_id="g", fingerprint="fp",
        concepts=[concept], edges=[],
    )
    store.set_communities(run_fingerprint="cfp", assignments={"concept:a": "community:x"})
    store.put_community_summary(
        CommunitySummary(
            community_id="community:x", label="Theme", summary="A synthesized cross-book summary.",
            agreements=(), divergences=(), concept_ids=("concept:a",),
            source_chunk_ids=(seed_chunk.chunk_id,), source_ids=(seed_chunk.meta.source_id,),
            model="m", prompt_version="v1", created_at="now",
            vector=tuple(embedder.embed_query(query)),
        )
    )

    seed = Retrieved(chunk=seed_chunk, score=0.1)  # deliberately low, so the summary must rank above it
    inner = _FixedRetriever([seed])
    retriever = ConceptRetriever(inner, store, corpus, embedder, summary_slots=1)

    results = retriever.retrieve(query, k=5)

    summary_result = next(
        (r for r in results if r.chunk.meta.extra.get("ragcore_kind") == "community_summary"), None
    )
    assert summary_result is not None
    assert summary_result.chunk.chunk_id == "community:x"
    assert summary_result.chunk.meta.source_type == "synthetic"
    assert summary_result.chunk.text == "A synthesized cross-book summary."
    assert summary_result.chunk.meta.extra["concept_ids"] == ["concept:a"]


def test_respects_k(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    all_chunks = corpus.store.all_chunks()
    seeds = [Retrieved(chunk=c, score=1.0 - i * 0.1) for i, c in enumerate(all_chunks)]

    store = SqliteConceptStore(":memory:")
    retriever = ConceptRetriever(_FixedRetriever(seeds), store, corpus, FakeEmbedder(dim=8))

    assert retriever.retrieve("q", k=0) == []
    results = retriever.retrieve("q", k=1)
    assert len(results) == 1
