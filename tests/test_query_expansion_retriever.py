"""QueryExpansionRetriever: retrieves the original query plus every
transform.expand() variant, fuses result sets by RRF. Pure Python
doubles — no embedder, no LLM, no store.
"""

from __future__ import annotations

from ragcore.retrieval.query_expansion import QueryExpansionRetriever
from ragcore.types import Chunk, ChunkMeta, Retrieved


def _chunk(chunk_id: str, text: str = "text") -> Chunk:
    meta = ChunkMeta(
        source_id="src", source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=None, section=None,
        page_start=None, page_end=None, chunk_index=0, context_group_id="g",
        char_start=0, char_end=len(text), token_count=1,
    )
    return Chunk(chunk_id=chunk_id, text=text, meta=meta)


class FakeInnerRetriever:
    """Returns a scripted result list per exact query string, so a test
    can control precisely what each variant 'finds'."""

    def __init__(self, results_by_query: dict[str, list[Retrieved]]) -> None:
        self._results_by_query = results_by_query
        self.calls: list[str] = []

    def retrieve(self, query, *, k=10, filters=None):
        self.calls.append(query)
        return self._results_by_query.get(query, [])[:k]


class FakeTransform:
    def __init__(self, variants: list[str]) -> None:
        self._variants = variants

    def expand(self, query: str) -> list[str]:
        return self._variants


def test_always_retrieves_the_original_query_plus_every_variant() -> None:
    inner = FakeInnerRetriever({})
    retriever = QueryExpansionRetriever(inner, FakeTransform(["variant a", "variant b"]))

    retriever.retrieve("original")

    assert inner.calls == ["original", "variant a", "variant b"]


def test_a_chunk_found_by_multiple_variants_ranks_above_one_found_by_one() -> None:
    c1, c2 = _chunk("c1"), _chunk("c2")
    inner = FakeInnerRetriever(
        {
            "original": [Retrieved(chunk=c1, score=0.5)],
            "hyde passage": [Retrieved(chunk=c1, score=0.6), Retrieved(chunk=c2, score=0.9)],
        }
    )
    retriever = QueryExpansionRetriever(inner, FakeTransform(["hyde passage"]))

    results = retriever.retrieve("original", k=10)

    # c1 found by both variants (rank 1 in each) should outrank c2 (found
    # by only one variant, even at a higher raw score), per RRF.
    assert [r.chunk.chunk_id for r in results] == ["c1", "c2"]


def test_each_result_keeps_its_own_best_real_score_not_an_rrf_score() -> None:
    c1 = _chunk("c1")
    inner = FakeInnerRetriever(
        {
            "original": [Retrieved(chunk=c1, score=0.3)],
            "variant": [Retrieved(chunk=c1, score=0.8)],
        }
    )
    retriever = QueryExpansionRetriever(inner, FakeTransform(["variant"]))

    [result] = retriever.retrieve("original")

    assert result.score == 0.8  # best of the two, not a tiny RRF score


def test_contributing_variants_are_stamped_into_extra() -> None:
    c1 = _chunk("c1")
    inner = FakeInnerRetriever(
        {
            "original": [Retrieved(chunk=c1, score=0.5)],
            "variant": [Retrieved(chunk=c1, score=0.6)],
        }
    )
    retriever = QueryExpansionRetriever(inner, FakeTransform(["variant"]))

    [result] = retriever.retrieve("original")

    assert result.chunk.meta.extra["query_expansion_variants"] == ("original", "variant")


def test_respects_k_on_the_final_merged_result() -> None:
    chunks = [_chunk(f"c{i}") for i in range(5)]
    inner = FakeInnerRetriever(
        {"original": [Retrieved(chunk=c, score=1.0 - i * 0.1) for i, c in enumerate(chunks)]}
    )
    retriever = QueryExpansionRetriever(inner, FakeTransform([]))

    results = retriever.retrieve("original", k=2)

    assert len(results) == 2


def test_k_zero_returns_empty_without_calling_inner() -> None:
    inner = FakeInnerRetriever({})
    retriever = QueryExpansionRetriever(inner, FakeTransform(["variant"]))

    assert retriever.retrieve("original", k=0) == []
    assert inner.calls == []


def test_no_variants_falls_back_to_just_the_original_query() -> None:
    c1 = _chunk("c1")
    inner = FakeInnerRetriever({"original": [Retrieved(chunk=c1, score=0.7)]})
    retriever = QueryExpansionRetriever(inner, FakeTransform([]))

    results = retriever.retrieve("original")

    assert inner.calls == ["original"]
    assert [r.chunk.chunk_id for r in results] == ["c1"]
