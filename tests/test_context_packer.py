"""pack_context unit tests — dedupe, parent/neighbor expansion, token
budgeting, and document-order presentation
(ragcore/generation/context_packer.py).
"""

from __future__ import annotations

from ragcore.chunking.tokenizer import count_tokens
from ragcore.corpus import Corpus
from ragcore.generation.context_packer import pack_context
from ragcore.types import Chunk, ChunkMeta, Retrieved
from tests.fakes import FakeEmbedder, InMemoryStore


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source_id: str = "src",
    chunk_index: int = 0,
    chapter_index: int | None = 0,
    char_start: int = 0,
) -> Chunk:
    meta = ChunkMeta(
        source_id=source_id, source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=chapter_index, section=None,
        page_start=None, page_end=None, chunk_index=chunk_index, context_group_id="g",
        char_start=char_start, char_end=char_start + len(text), token_count=count_tokens(text),
    )
    return Chunk(chunk_id=chunk_id, text=text, meta=meta)


def test_dedupes_by_chunk_id_keeping_the_first_occurrence() -> None:
    c1 = _chunk("c1", "first version")
    c1_dup = _chunk("c1", "should be ignored")
    packed = pack_context(
        [Retrieved(chunk=c1, score=0.9), Retrieved(chunk=c1_dup, score=0.5)], max_tokens=1000
    )
    assert len(packed.kept) == 1
    assert packed.kept[0].chunk.text == "first version"


def test_token_budget_drops_lowest_relevance_first() -> None:
    big_text = "word " * 50  # well over a tiny budget
    high = Retrieved(chunk=_chunk("c1", big_text, chunk_index=0), score=0.9)
    low = Retrieved(chunk=_chunk("c2", big_text, chunk_index=1), score=0.1)

    budget = count_tokens(big_text) + 5  # room for exactly one
    packed = pack_context([high, low], max_tokens=budget)

    assert [r.chunk.chunk_id for r in packed.kept] == ["c1"]
    assert [r.chunk.chunk_id for r in packed.dropped] == ["c2"]


def test_reserve_tokens_shrinks_the_effective_budget() -> None:
    text = "word " * 20
    r = Retrieved(chunk=_chunk("c1", text), score=0.9)
    tokens = count_tokens(text)

    fits = pack_context([r], max_tokens=tokens + 5, reserve_tokens=0)
    assert len(fits.kept) == 1

    doesnt_fit = pack_context([r], max_tokens=tokens + 5, reserve_tokens=10)
    assert len(doesnt_fit.kept) == 0
    assert len(doesnt_fit.dropped) == 1


def test_total_tokens_reflects_only_the_kept_set() -> None:
    kept_text = "short"
    dropped_text = "word " * 100
    kept = Retrieved(chunk=_chunk("c1", kept_text), score=0.9)
    dropped = Retrieved(chunk=_chunk("c2", dropped_text), score=0.1)

    packed = pack_context([kept, dropped], max_tokens=count_tokens(kept_text) + 2)

    assert packed.total_tokens == count_tokens(kept_text)


def test_kept_set_is_ordered_by_document_position_not_relevance() -> None:
    # c2 is more relevant (higher score) but appears earlier in the
    # source (lower chunk_index/char_start) than c1.
    c1 = _chunk("c1", "later passage", chunk_index=5, char_start=500)
    c2 = _chunk("c2", "earlier passage", chunk_index=0, char_start=0)
    packed = pack_context(
        [Retrieved(chunk=c1, score=0.9), Retrieved(chunk=c2, score=0.4)], max_tokens=1000
    )

    assert [r.chunk.chunk_id for r in packed.kept] == ["c2", "c1"]  # document order, not score order


def test_ordering_groups_by_source_then_chapter_then_position() -> None:
    a = _chunk("a", "x", source_id="book1", chapter_index=1, char_start=100)
    b = _chunk("b", "x", source_id="book1", chapter_index=0, char_start=0)
    c = _chunk("c", "x", source_id="book0", chapter_index=5, char_start=0)
    packed = pack_context(
        [Retrieved(chunk=a, score=0.9), Retrieved(chunk=b, score=0.5), Retrieved(chunk=c, score=0.1)],
        max_tokens=1000,
    )
    assert [r.chunk.chunk_id for r in packed.kept] == ["c", "b", "a"]  # book0 before book1, then by chapter


def test_neighbor_expansion_fetches_adjacent_chunks_from_the_same_source() -> None:
    store = InMemoryStore(dim=4)
    embedder = FakeEmbedder(dim=4)
    corpus = Corpus(store, embedder)
    source_id = "sharedsrc"
    chunks = [
        Chunk(chunk_id=f"{source_id}:{i}", text=f"chunk {i}", vector=[0.0] * 4, meta=ChunkMeta(
            source_id=source_id, source_path="p", source_title="t", source_type="md",
            section_path=(), chapter_title=None, chapter_index=0, section=None,
            page_start=None, page_end=None, chunk_index=i, context_group_id="g",
            char_start=i * 10, char_end=i * 10 + 7, token_count=2,
        ))
        for i in range(5)
    ]
    store.upsert(chunks)

    hit = Retrieved(chunk=chunks[2], score=0.9)
    packed = pack_context([hit], max_tokens=1000, corpus=corpus, expand_neighbors=1)

    ids = [r.chunk.chunk_id for r in packed.kept]
    assert ids == [f"{source_id}:1", f"{source_id}:2", f"{source_id}:3"]


def test_neighbor_expansion_respects_source_boundaries() -> None:
    store = InMemoryStore(dim=4)
    corpus = Corpus(store, FakeEmbedder(dim=4))
    source_id = "shortsrc"
    chunk0 = Chunk(chunk_id=f"{source_id}:0", text="only chunk", vector=[0.0] * 4, meta=ChunkMeta(
        source_id=source_id, source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=0, section=None,
        page_start=None, page_end=None, chunk_index=0, context_group_id="g",
        char_start=0, char_end=10, token_count=2,
    ))
    store.upsert([chunk0])

    hit = Retrieved(chunk=chunk0, score=0.9)
    packed = pack_context([hit], max_tokens=1000, corpus=corpus, expand_neighbors=2)

    assert [r.chunk.chunk_id for r in packed.kept] == [f"{source_id}:0"]  # no neighbors exist


def test_expansion_chunk_scores_below_the_hit_it_expands_for_budget_priority() -> None:
    store = InMemoryStore(dim=4)
    corpus = Corpus(store, FakeEmbedder(dim=4))
    source_id = "src"
    big_text = "word " * 30
    chunks = [
        Chunk(chunk_id=f"{source_id}:{i}", text=big_text, vector=[0.0] * 4, meta=ChunkMeta(
            source_id=source_id, source_path="p", source_title="t", source_type="md",
            section_path=(), chapter_title=None, chapter_index=0, section=None,
            page_start=None, page_end=None, chunk_index=i, context_group_id="g",
            char_start=i * 200, char_end=i * 200 + len(big_text), token_count=count_tokens(big_text),
        ))
        for i in range(2)
    ]
    store.upsert(chunks)

    # Enough budget for exactly one chunk's worth of tokens — the real
    # hit (chunks[0]) must win over its own neighbor expansion
    # (chunks[1]) when both can't fit.
    budget = count_tokens(big_text) + 5
    hit = Retrieved(chunk=chunks[0], score=0.9)
    packed = pack_context([hit], max_tokens=budget, corpus=corpus, expand_neighbors=1)

    assert [r.chunk.chunk_id for r in packed.kept] == [f"{source_id}:0"]


def test_no_expansion_without_corpus_even_if_expand_neighbors_set() -> None:
    c1 = _chunk("c1", "text")
    packed = pack_context([Retrieved(chunk=c1, score=0.9)], max_tokens=1000, expand_neighbors=2)
    assert [r.chunk.chunk_id for r in packed.kept] == ["c1"]


def test_empty_context_returns_empty_packed_context() -> None:
    packed = pack_context([], max_tokens=1000)
    assert packed.kept == []
    assert packed.dropped == []
    assert packed.total_tokens == 0
