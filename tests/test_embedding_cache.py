"""Corpus-level integration tests for the opt-in embedding_cache=...
(ragcore.embedding.cache) — content-hash + model-version caching at
Corpus._embed_groups()'s ChunkGroup granularity.

CountingEmbedder records each embed_groups() call's full `groups`
argument (every miss-group in one call is batched together, same as the
uncached path's batching — see Corpus._embed_groups_with_cache), so
assertions check the *groups actually sent*, not a one-call-per-group
count.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

from ragcore.corpus import Corpus
from ragcore.embedding.cache import SqliteEmbeddingCache
from ragcore.types import Chunk, Vector
from tests.fakes import FakeEmbedder, InMemoryStore


def _assert_vectors_approx_equal(
    a: dict[str, Vector | None], b: dict[str, Vector | None]
) -> None:
    """The cache round-trips vectors through a float32 pack (see
    ragcore/embedding/cache.py) — a vector read back from it is never
    bit-for-bit identical to one computed fresh in float64, by design."""
    assert a.keys() == b.keys()
    for chunk_id, va in a.items():
        vb = b[chunk_id]
        assert va is not None and vb is not None
        assert len(va) == len(vb)
        assert all(abs(x - y) < 1e-6 for x, y in zip(va, vb))


def _vectors_by_chunk_id(chunks: list[Chunk]) -> dict[str, Vector | None]:
    return {c.chunk_id: c.vector for c in chunks}


class CountingEmbedder:
    """Wraps a FakeEmbedder, recording every embed_groups() call's groups
    so a test can assert exactly what was (and wasn't) re-embedded."""

    def __init__(self, base: FakeEmbedder) -> None:
        self._base = base
        self.name = base.name
        self.dim = base.dim
        self.calls: list[list[list[str]]] = []

    def embed_groups(
        self,
        groups: Sequence[Sequence[str]],
        *,
        input_type: Literal["document", "query", None] = "document",
    ) -> list[list[Vector]]:
        self.calls.append([list(g) for g in groups])
        return self._base.embed_groups(groups, input_type=input_type)

    def embed_query(self, text: str) -> Vector:
        return self._base.embed_query(text)

    def groups_embedded(self) -> int:
        """Total ChunkGroups sent across every embed_groups() call —
        the thing tests actually care about, independent of how many
        miss-groups happened to land in the same request."""
        return sum(len(call) for call in self.calls)


def test_reingesting_identical_content_skips_every_embed_call(
    tmp_path: Path, md_book: Path
) -> None:
    embedder = CountingEmbedder(FakeEmbedder(dim=8))
    store = InMemoryStore(dim=8)
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite3")
    corpus = Corpus(store, embedder, embedding_cache=cache)

    corpus.ingest(md_book)
    assert embedder.groups_embedded() == 2  # two chapters, both cache misses first time

    corpus.ingest(md_book)  # replace=True, identical bytes -> identical chunk text
    assert embedder.groups_embedded() == 2  # no new embedding on the second ingest
    assert len(embedder.calls) == 1  # confirms no call was made at all, not just an empty one


def test_cache_is_shared_across_two_different_corpus_instances(
    tmp_path: Path, md_book: Path
) -> None:
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite3")

    embedder1 = CountingEmbedder(FakeEmbedder(dim=8))
    corpus1 = Corpus(InMemoryStore(dim=8), embedder1, embedding_cache=cache)
    corpus1.ingest(md_book)
    assert embedder1.groups_embedded() == 2

    embedder2 = CountingEmbedder(FakeEmbedder(dim=8))  # same model name/dim, different instance
    corpus2 = Corpus(InMemoryStore(dim=8), embedder2, embedding_cache=cache)
    corpus2.ingest(md_book)
    assert embedder2.calls == []  # fully served from the shared cache, no request at all

    # and the vectors are actually the same, not just "no call was made"
    _assert_vectors_approx_equal(
        _vectors_by_chunk_id(corpus1.store.all_chunks()),
        _vectors_by_chunk_id(corpus2.store.all_chunks()),
    )


def test_a_partial_hit_within_a_group_forces_the_whole_group_to_reembed(
    tmp_path: Path, md_book: Path
) -> None:
    """Contextualized embedding needs every chunk in a group present
    together — a group is only skipped when every one of its chunks is
    already cached, never a subset."""
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite3")
    embedder = CountingEmbedder(FakeEmbedder(dim=8))
    store = InMemoryStore(dim=8)
    corpus = Corpus(store, embedder, embedding_cache=cache)

    corpus.ingest(md_book)
    assert embedder.groups_embedded() == 2

    # Evict exactly one chunk's cache row from chapter 1's group (2 chunks).
    [source] = store.list_sources()
    evicted_chunk = store.get([f"{source.source_id}:0"])[0]
    key = Corpus._cache_key(evicted_chunk.text, embedder.name, embedder.dim)
    cache._conn.execute(  # test-only: reach into the store to simulate partial eviction
        "DELETE FROM embedding_cache WHERE content_hash = ? AND model_name = ? AND dim = ?",
        key,
    )

    corpus.ingest(md_book)
    # chapter 1's group (now partially missing) must be fully re-embedded;
    # chapter 2's group (still fully cached) must not trigger any call.
    assert len(embedder.calls) == 2  # one new call, for chapter 1's group only
    [second_call] = embedder.calls[1:]
    assert len(second_call) == 1  # exactly chapter 1's one group
    assert len(second_call[0]) == 2  # with both of its chunks, not just the evicted one


def test_switching_models_does_not_disturb_the_old_models_cache_entries(
    tmp_path: Path, md_book: Path
) -> None:
    """Zero-downtime re-embedding migration: a model swap is a guaranteed
    cache miss (different model name), and the old model's rows stay
    exactly as they were."""
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite3")

    old_embedder = CountingEmbedder(FakeEmbedder(dim=8))
    old_embedder.name = "model-v1"
    corpus_old = Corpus(InMemoryStore(dim=8), old_embedder, embedding_cache=cache)
    corpus_old.ingest(md_book)
    old_vectors = _vectors_by_chunk_id(corpus_old.store.all_chunks())

    new_embedder = CountingEmbedder(FakeEmbedder(dim=8))
    new_embedder.name = "model-v2"
    corpus_new = Corpus(InMemoryStore(dim=8), new_embedder, embedding_cache=cache)
    corpus_new.ingest(md_book)
    assert new_embedder.groups_embedded() == 2  # new model: guaranteed cache miss

    # old model's entries are untouched — a Corpus still on the old model
    # would still be fully served from cache with no re-embedding.
    old_embedder_again = CountingEmbedder(FakeEmbedder(dim=8))
    old_embedder_again.name = "model-v1"
    corpus_old_again = Corpus(InMemoryStore(dim=8), old_embedder_again, embedding_cache=cache)
    corpus_old_again.ingest(md_book)
    assert old_embedder_again.calls == []
    _assert_vectors_approx_equal(
        _vectors_by_chunk_id(corpus_old_again.store.all_chunks()), old_vectors
    )


def test_without_embedding_cache_every_ingest_reembeds(tmp_path: Path, md_book: Path) -> None:
    embedder = CountingEmbedder(FakeEmbedder(dim=8))
    corpus = Corpus(InMemoryStore(dim=8), embedder)  # no embedding_cache

    corpus.ingest(md_book)
    corpus.ingest(md_book)
    assert len(embedder.calls) == 2  # every ingest re-embeds, no cache to skip it
    assert embedder.groups_embedded() == 4
