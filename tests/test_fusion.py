"""rrf_merge unit tests — the shared reciprocal rank fusion helper
(ragcore/retrieval/fusion.py), extracted from and exercised through
QueryExpansionRetriever's own tests too (tests/test_query_expansion_retriever.py)."""

from __future__ import annotations

from ragcore.retrieval.fusion import rrf_merge
from ragcore.types import Chunk, ChunkMeta, Retrieved


def _chunk(chunk_id: str, text: str = "text") -> Chunk:
    meta = ChunkMeta(
        source_id="src", source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=None, section=None,
        page_start=None, page_end=None, chunk_index=0, context_group_id="g",
        char_start=0, char_end=len(text), token_count=1,
    )
    return Chunk(chunk_id=chunk_id, text=text, meta=meta)


def test_a_chunk_found_by_more_lists_ranks_above_one_found_by_fewer() -> None:
    c1, c2 = _chunk("c1"), _chunk("c2")
    merged = rrf_merge(
        [
            ("a", [Retrieved(chunk=c1, score=0.5)]),
            ("b", [Retrieved(chunk=c1, score=0.6), Retrieved(chunk=c2, score=0.9)]),
        ],
        k=10,
    )
    assert [r.chunk.chunk_id for r in merged] == ["c1", "c2"]


def test_each_result_keeps_its_own_best_score_not_a_raw_rrf_score() -> None:
    c1 = _chunk("c1")
    [result] = rrf_merge(
        [("a", [Retrieved(chunk=c1, score=0.3)]), ("b", [Retrieved(chunk=c1, score=0.8)])],
        k=10,
    )
    assert result.score == 0.8


def test_contributing_labels_are_stamped_under_the_given_extra_key() -> None:
    c1 = _chunk("c1")
    [result] = rrf_merge(
        [("mode-a", [Retrieved(chunk=c1, score=0.5)]), ("mode-b", [Retrieved(chunk=c1, score=0.6)])],
        k=10,
        extra_key="routed_modes",
    )
    assert result.chunk.meta.extra["routed_modes"] == ("mode-a", "mode-b")
    assert "rrf_sources" not in result.chunk.meta.extra


def test_default_extra_key_is_rrf_sources() -> None:
    c1 = _chunk("c1")
    [result] = rrf_merge([("a", [Retrieved(chunk=c1, score=0.5)])], k=10)
    assert result.chunk.meta.extra["rrf_sources"] == ("a",)


def test_respects_k() -> None:
    chunks = [_chunk(f"c{i}") for i in range(5)]
    merged = rrf_merge(
        [("a", [Retrieved(chunk=c, score=1.0 - i * 0.1) for i, c in enumerate(chunks)])], k=2
    )
    assert len(merged) == 2


def test_empty_input_returns_empty() -> None:
    assert rrf_merge([], k=10) == []


def test_a_list_with_no_results_contributes_nothing() -> None:
    c1 = _chunk("c1")
    merged = rrf_merge([("a", [Retrieved(chunk=c1, score=0.5)]), ("b", [])], k=10)
    assert [r.chunk.chunk_id for r in merged] == ["c1"]
    assert merged[0].chunk.meta.extra["rrf_sources"] == ("a",)


def _rank_of(chunk_id: str, results: list[Retrieved]) -> int:
    return next(i for i, r in enumerate(results) if r.chunk.chunk_id == chunk_id)


def test_rrf_k_parameter_changes_the_fusion_curve() -> None:
    """Sanity check that rrf_k is actually wired through, not just
    accepted and ignored: c1 is rank 1 in list "a" only (RRF score
    1/(k+1)); c2 is rank 5 or 6 in *both* lists (RRF score
    1/(k+6) + 1/(k+5), using distinct filler chunks per list so the
    filler itself never competes for the top ranks). A small rrf_k makes
    a single top rank dominate; a large rrf_k flattens the curve enough
    that c2's two appearances overtake c1's one top rank."""
    c1, c2 = _chunk("c1"), _chunk("c2")
    filler_a = [_chunk(f"fa{i}") for i in range(4)]
    filler_b = [_chunk(f"fb{i}") for i in range(4)]
    list_a = (
        [Retrieved(chunk=c1, score=0.9)]
        + [Retrieved(chunk=f, score=0.5) for f in filler_a]
        + [Retrieved(chunk=c2, score=0.1)]
    )
    list_b = [Retrieved(chunk=f, score=0.5) for f in filler_b] + [Retrieved(chunk=c2, score=0.1)]
    lists = [("a", list_a), ("b", list_b)]

    tiny_rrf_k = rrf_merge(lists, k=10, rrf_k=1)
    large_rrf_k = rrf_merge(lists, k=10, rrf_k=1000)

    assert _rank_of("c1", tiny_rrf_k) < _rank_of("c2", tiny_rrf_k)
    assert _rank_of("c2", large_rrf_k) < _rank_of("c1", large_rrf_k)
