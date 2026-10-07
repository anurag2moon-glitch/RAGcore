"""CachingRetriever: cache hit avoids calling the inner retriever again,
a different (query, k, filters) is a different cache key, and the
tenant_id tag flows from filters through to the cache's own tenant-scoped
invalidation.
"""

from __future__ import annotations

from pathlib import Path

from ragcore.caching.memory import InMemoryResultCache
from ragcore.caching.retriever import CachingRetriever
from ragcore.corpus import Corpus
from ragcore.retrieval.vector import VectorRetriever
from tests.fakes import FakeEmbedder, InMemoryStore


class _CountingRetriever:
    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls = 0

    def retrieve(self, query, *, k=10, filters=None):
        self.calls += 1
        return self._inner.retrieve(query, k=k, filters=filters)


def _corpus(md_book: Path) -> Corpus:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    return corpus


def test_a_cache_hit_does_not_call_the_inner_retriever_again(md_book: Path) -> None:
    corpus = _corpus(md_book)
    inner = _CountingRetriever(VectorRetriever(corpus.embedder, corpus.store))
    cache = InMemoryResultCache()
    retriever = CachingRetriever(inner, cache)

    r1 = retriever.retrieve("chunking", k=3)
    r2 = retriever.retrieve("chunking", k=3)

    assert inner.calls == 1
    assert [r.chunk.chunk_id for r in r1] == [r.chunk.chunk_id for r in r2]


def test_a_different_k_is_a_different_cache_key(md_book: Path) -> None:
    corpus = _corpus(md_book)
    inner = _CountingRetriever(VectorRetriever(corpus.embedder, corpus.store))
    retriever = CachingRetriever(inner, InMemoryResultCache())

    retriever.retrieve("chunking", k=3)
    retriever.retrieve("chunking", k=5)

    assert inner.calls == 2


def test_a_different_filters_dict_is_a_different_cache_key(md_book: Path) -> None:
    corpus = _corpus(md_book)
    inner = _CountingRetriever(VectorRetriever(corpus.embedder, corpus.store))
    retriever = CachingRetriever(inner, InMemoryResultCache())

    retriever.retrieve("chunking", k=5, filters={"tenant_id": "tenant-a"})
    retriever.retrieve("chunking", k=5, filters={"tenant_id": "tenant-b"})

    assert inner.calls == 2


def test_tenant_id_filter_tags_the_cache_entry_for_targeted_invalidation(md_book: Path) -> None:
    corpus = _corpus(md_book)
    inner = _CountingRetriever(VectorRetriever(corpus.embedder, corpus.store))
    cache = InMemoryResultCache()
    retriever = CachingRetriever(inner, cache)

    retriever.retrieve("chunking", k=3, filters={"tenant_id": "tenant-a"})
    assert len(cache) == 1

    removed = cache.invalidate(tenant_id="tenant-a")

    assert removed == 1
    retriever.retrieve("chunking", k=3, filters={"tenant_id": "tenant-a"})
    assert inner.calls == 2  # cache miss after invalidation, re-queried


def test_invalidating_a_different_tenant_does_not_evict_this_ones_entry(md_book: Path) -> None:
    corpus = _corpus(md_book)
    inner = _CountingRetriever(VectorRetriever(corpus.embedder, corpus.store))
    cache = InMemoryResultCache()
    retriever = CachingRetriever(inner, cache)

    retriever.retrieve("chunking", k=3, filters={"tenant_id": "tenant-a"})
    cache.invalidate(tenant_id="tenant-b")
    retriever.retrieve("chunking", k=3, filters={"tenant_id": "tenant-a"})

    assert inner.calls == 1  # still a hit — tenant-b's invalidation didn't touch tenant-a
