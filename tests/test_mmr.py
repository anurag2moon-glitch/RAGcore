"""MMRReranker composing with a fixed inner Retriever — no embeddings, no
network. Verifies the per-source cap (and its soft relaxation, which must
never return fewer results than the inner retriever would have), the
lambda_=1.0 degenerate case, the len(pool) <= k no-op, and each
on_missing_vectors mode.
"""

from __future__ import annotations

from collections import Counter

import pytest

from ragcore.errors import RagcoreError
from ragcore.retrieval.mmr import MMRReranker
from ragcore.types import Chunk, ChunkMeta, Filter, Retrieved


class _FixedRetriever:
    """Retriever stub returning a pre-set list of Retrieved for any query —
    same shape as test_graph_retrieval.py's."""

    def __init__(self, results: list[Retrieved]) -> None:
        self._results = results
        self.calls: list[tuple[str, int, Filter | None]] = []

    def retrieve(self, query: str, *, k: int = 10, filters: Filter | None = None) -> list[Retrieved]:
        self.calls.append((query, k, filters))
        return self._results[:k]


def _mk(chunk_id: str, source_id: str, score: float, vector, text: str = "some words here") -> Retrieved:
    meta = ChunkMeta(
        source_id=source_id, source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=None, section=None,
        page_start=None, page_end=None, chunk_index=0, context_group_id="g",
        char_start=0, char_end=len(text), token_count=len(text) // 4,
    )
    return Retrieved(chunk=Chunk(chunk_id=chunk_id, text=text, meta=meta, vector=vector), score=score)


def _mk_summary(chunk_id: str, score: float, vector) -> Retrieved:
    r = _mk(chunk_id, f"community:{chunk_id}", score, vector)
    r.chunk.meta.extra["ragcore_kind"] = "community_summary"
    return r


_POOL = [
    _mk("a1", "A", 0.95, [1.0, 0.0]),
    _mk("a2", "A", 0.94, [0.99, 0.01]),
    _mk("a3", "A", 0.93, [0.98, 0.02]),
    _mk("a4", "A", 0.92, [0.97, 0.03]),
    _mk("b1", "B", 0.5, [0.0, 1.0]),
    _mk("b2", "B", 0.4, [-1.0, 0.0]),
]


def test_len_pool_at_most_k_is_a_no_op() -> None:
    retriever = MMRReranker(_FixedRetriever(_POOL[:3]), fetch_multiplier=3)
    results = retriever.retrieve("q", k=10)
    assert results == _POOL[:3]


def test_k_le_0_returns_empty() -> None:
    retriever = MMRReranker(_FixedRetriever(_POOL), fetch_multiplier=3)
    assert retriever.retrieve("q", k=0) == []


def test_lambda_one_degenerates_to_pure_relevance_order() -> None:
    retriever = MMRReranker(_FixedRetriever(_POOL), lambda_=1.0, per_source_cap=None, fetch_multiplier=3)
    results = retriever.retrieve("q", k=4)
    assert [r.chunk.chunk_id for r in results] == ["a1", "a2", "a3", "a4"]


def test_per_source_cap_is_respected_when_enough_diverse_candidates_exist() -> None:
    retriever = MMRReranker(_FixedRetriever(_POOL), lambda_=0.5, per_source_cap=2, fetch_multiplier=3)
    results = retriever.retrieve("q", k=4)
    counts = Counter(r.chunk.meta.source_id for r in results)
    assert counts["A"] <= 2
    assert len(results) == 4


def test_per_source_cap_relaxes_rather_than_returning_fewer_than_k() -> None:
    """Only source A has candidates at all — a hard cap would return fewer
    than k results; the cap must relax instead, since diversity is traded
    against ordering, never against recall."""
    only_a = [r for r in _POOL if r.chunk.meta.source_id == "A"]
    retriever = MMRReranker(_FixedRetriever(only_a), lambda_=0.5, per_source_cap=1, fetch_multiplier=3)
    results = retriever.retrieve("q", k=4)
    assert len(results) == 4


def test_max_summaries_caps_synthetic_chunks_independently_of_source_cap() -> None:
    # Pool deliberately larger than k so the len(pool) <= k no-op path
    # (which bypasses every cap, by design) never triggers.
    summaries = [_mk_summary(f"s{i}", 0.99, [1.0, 0.0]) for i in range(3)]
    pool = summaries + [r for r in _POOL if r.chunk.meta.source_id == "B"] + [_mk("extra", "C", 0.1, [0.0, -1.0])]
    retriever = MMRReranker(_FixedRetriever(pool), lambda_=1.0, max_summaries=1, fetch_multiplier=3)
    results = retriever.retrieve("q", k=3)
    n_summaries = sum(1 for r in results if r.chunk.meta.extra.get("ragcore_kind") == "community_summary")
    assert n_summaries == 1


def test_passthrough_returns_pool_unranked_when_any_vector_is_missing() -> None:
    pool = _POOL[:3] + [_mk("no_vec", "C", 0.6, None)]
    retriever = MMRReranker(_FixedRetriever(pool), on_missing_vectors="passthrough", fetch_multiplier=3)
    results = retriever.retrieve("q", k=2)
    assert [r.chunk.chunk_id for r in results] == [r.chunk.chunk_id for r in pool[:2]]


def test_error_mode_raises_when_a_vector_is_missing() -> None:
    pool = _POOL[:3] + [_mk("no_vec", "C", 0.6, None)]
    retriever = MMRReranker(_FixedRetriever(pool), on_missing_vectors="error", fetch_multiplier=3)
    with pytest.raises(RagcoreError, match="on_missing_vectors"):
        retriever.retrieve("q", k=2)


def test_lexical_mode_still_returns_k_results_with_missing_vectors() -> None:
    pool = _POOL[:3] + [_mk("no_vec", "C", 0.6, None, text="entirely different words")]
    retriever = MMRReranker(_FixedRetriever(pool), on_missing_vectors="lexical", fetch_multiplier=3)
    results = retriever.retrieve("q", k=4)
    assert len(results) == 4
