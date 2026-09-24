"""HybridRetriever: reciprocal rank fusion of a vector channel and a lexical
channel, weight configurability, and — the specific correctness property
this component exists to get right — that fused results are scored on the
same cosine-derived scale every other retrieval layer uses, never a raw RRF
magnitude (see ragcore/retrieval/hybrid.py's module docstring for the
concrete failure mode this test guards against).
"""

from __future__ import annotations

from pathlib import Path

from ragcore.corpus import Corpus
from ragcore.lexical.fts5 import Fts5LexicalStore
from ragcore.retrieval.hybrid import HybridRetriever, _source_ids_from_filter
from tests.fakes import FakeEmbedder, InMemoryStore


def _corpus_with_chunks(md_book: Path) -> Corpus:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    return corpus


def _index_all(corpus: Corpus, lexical: Fts5LexicalStore) -> None:
    for source in corpus.sources():
        lexical.index(corpus.store.get_by_source(source.source_id))


def test_fused_scores_are_on_the_cosine_scale_not_raw_rrf_magnitude(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    lexical = Fts5LexicalStore(":memory:")
    _index_all(corpus, lexical)

    retriever = HybridRetriever(corpus.embedder, corpus.store, lexical, corpus)
    results = retriever.retrieve("contextualized retrieval", k=5)

    assert results
    # A raw RRF score at rrf_k=60 is at most 1/(60+1) ~= 0.0164 for a single
    # channel, or ~0.033 if a hit ranks #1 in both channels combined —
    # nowhere near the 0.33-1.0 cosine-score range every other layer uses.
    for r in results:
        assert 0.0 <= r.score <= 1.0
        assert r.score > 0.05, (
            f"score {r.score} looks like a raw RRF magnitude, not a cosine_score() result"
        )


def test_lexical_only_hit_is_hydrated_and_included(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    lexical = Fts5LexicalStore(":memory:")
    _index_all(corpus, lexical)

    all_chunks = corpus.store.all_chunks()
    # Find a real distinctive word from a real chunk's text to search on.
    target = all_chunks[0]
    word = next(w for w in target.text.split() if len(w) > 5)

    retriever = HybridRetriever(corpus.embedder, corpus.store, lexical, corpus, channel_multiplier=1)
    results = retriever.retrieve(word, k=10)

    chunk_ids = {r.chunk.chunk_id for r in results}
    assert target.chunk_id in chunk_ids


def test_rrf_score_and_rank_are_stamped_for_debuggability(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    lexical = Fts5LexicalStore(":memory:")
    _index_all(corpus, lexical)

    retriever = HybridRetriever(corpus.embedder, corpus.store, lexical, corpus)
    results = retriever.retrieve("structural chunking method", k=5)

    for r in results:
        assert "rrf_score" in r.chunk.meta.extra
        assert "rrf_rank" in r.chunk.meta.extra
        assert r.chunk.meta.extra["rrf_rank"] >= 1


def test_lexical_weight_zero_degenerates_to_vector_only_candidate_set(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    lexical = Fts5LexicalStore(":memory:")
    _index_all(corpus, lexical)

    vector_only = HybridRetriever(
        corpus.embedder, corpus.store, lexical, corpus, lexical_weight=0.0
    )
    results = vector_only.retrieve("chunking", k=len(corpus.store.all_chunks()))

    # With lexical_weight=0, a lexical-only hit contributes nothing to its
    # rrf_score beyond what the vector channel already gave it — it can
    # still appear (vector similarity alone may rank it), but a chunk that
    # is a lexical hit ONLY (never in the vector channel's fetch window)
    # must not be pulled in purely by keyword match when its weight is zero.
    for r in results:
        assert r.chunk.meta.extra["rrf_score"] > 0.0


def test_k_le_0_returns_empty(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    lexical = Fts5LexicalStore(":memory:")
    retriever = HybridRetriever(corpus.embedder, corpus.store, lexical, corpus)
    assert retriever.retrieve("anything", k=0) == []


def test_source_ids_from_filter_extracts_plain_equality_and_membership() -> None:
    assert _source_ids_from_filter(None) is None
    assert _source_ids_from_filter({}) is None
    assert _source_ids_from_filter({"source_id": "s1"}) == ["s1"]
    assert _source_ids_from_filter({"source_id": ["s1", "s2"]}) == ["s1", "s2"]
    assert _source_ids_from_filter({"source_id": {"eq": "s1"}}) is None
    assert _source_ids_from_filter({"source_id": "s1", "chapter_index": 1}) is None


def test_source_id_filter_is_pushed_down_to_the_lexical_channel(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    lexical = Fts5LexicalStore(":memory:")
    _index_all(corpus, lexical)
    source_id = corpus.sources()[0].source_id

    retriever = HybridRetriever(corpus.embedder, corpus.store, lexical, corpus)
    results = retriever.retrieve("chunking", k=20, filters={"source_id": source_id})

    assert all(r.chunk.meta.source_id == source_id for r in results)
