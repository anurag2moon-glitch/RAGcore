"""CrossEncoderReranker: stamps chunk.meta.extra["cross_encoder_score"]
without touching Retrieved.score, re-sorts by it, and composes correctly
with MMRReranker's all-or-nothing relevance-source switch. No network — a
deterministic fake Reranker stands in for VoyageReranker.
"""

from __future__ import annotations

from ragcore.reranking.base import RerankHit
from ragcore.retrieval.cross_encoder import CrossEncoderReranker
from ragcore.retrieval.mmr import MMRReranker
from ragcore.types import Chunk, ChunkMeta, Retrieved


def _mk(chunk_id: str, source_id: str, score: float, text: str, vector=None) -> Retrieved:
    meta = ChunkMeta(
        source_id=source_id, source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=None, section=None,
        page_start=None, page_end=None, chunk_index=0, context_group_id="g",
        char_start=0, char_end=len(text), token_count=1,
    )
    return Retrieved(chunk=Chunk(chunk_id=chunk_id, text=text, meta=meta, vector=vector), score=score)


class _FixedRetriever:
    def __init__(self, results: list[Retrieved]) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query, *, k=10, filters=None):
        self.calls.append((query, k))
        return self._results[:k]


class _FixedReranker:
    """Returns hand-set relevance scores per document text, independent of
    the input order — proves CrossEncoderReranker maps indices correctly."""

    model = "fixed"

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self._scores_by_text = scores_by_text
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query, documents, *, top_n=None):
        self.calls.append((query, list(documents)))
        hits = [
            RerankHit(index=i, relevance_score=self._scores_by_text[doc])
            for i, doc in enumerate(documents)
        ]
        return sorted(hits, key=lambda h: h.relevance_score, reverse=True)


def test_rescores_and_reorders_by_relevance_score() -> None:
    pool = [_mk("a", "s1", 0.9, "alpha"), _mk("b", "s1", 0.5, "beta"), _mk("c", "s1", 0.1, "gamma")]
    inner = _FixedRetriever(pool)
    reranker = _FixedReranker({"alpha": 0.1, "beta": 0.9, "gamma": 0.5})

    retriever = CrossEncoderReranker(inner, reranker, fetch_multiplier=1)
    results = retriever.retrieve("q", k=3)

    assert [r.chunk.chunk_id for r in results] == ["b", "c", "a"]


def test_does_not_overwrite_retrieved_score() -> None:
    pool = [_mk("a", "s1", 0.42, "alpha")]
    inner = _FixedRetriever(pool)
    reranker = _FixedReranker({"alpha": 0.99})

    results = CrossEncoderReranker(inner, reranker, fetch_multiplier=1).retrieve("q", k=1)

    assert results[0].score == 0.42  # unchanged
    assert results[0].chunk.meta.extra["cross_encoder_score"] == 0.99


def test_fetch_multiplier_controls_how_much_is_pulled_from_inner() -> None:
    pool = [_mk(f"c{i}", "s1", 1.0 - i * 0.1, f"text {i}") for i in range(10)]
    inner = _FixedRetriever(pool)
    reranker = _FixedReranker({f"text {i}": 1.0 - i * 0.1 for i in range(10)})

    CrossEncoderReranker(inner, reranker, fetch_multiplier=3).retrieve("q", k=2)

    assert inner.calls == [("q", 6)]  # k * fetch_multiplier


def test_k_le_0_returns_empty_without_calling_inner() -> None:
    inner = _FixedRetriever([_mk("a", "s1", 0.9, "alpha")])
    reranker = _FixedReranker({})

    assert CrossEncoderReranker(inner, reranker).retrieve("q", k=0) == []
    assert inner.calls == []


def test_mmr_prefers_cross_encoder_score_when_every_candidate_has_one() -> None:
    # Embedding-derived scores would rank a > b > c > d; cross-encoder scores
    # deliberately invert that — MMR must follow the cross-encoder ranking.
    # Pool deliberately larger than k so the len(pool) <= k no-op path never
    # triggers (that path bypasses every scoring decision, by design).
    pool = [
        _mk("a", "s1", 0.9, "text a", vector=[1.0, 0.0]),
        _mk("b", "s2", 0.6, "text b", vector=[0.0, 1.0]),
        _mk("c", "s3", 0.3, "text c", vector=[-1.0, 0.0]),
        _mk("d", "s4", 0.1, "text d", vector=[0.0, -1.0]),
    ]
    for i, r in enumerate(pool):
        r.chunk.meta.extra["cross_encoder_score"] = [0.1, 0.3, 0.6, 0.9][i]  # d > c > b > a

    class _Fixed:
        def retrieve(self, query, *, k=10, filters=None):
            return pool[:k]

    retriever = MMRReranker(_Fixed(), lambda_=1.0, per_source_cap=None, fetch_multiplier=2)
    results = retriever.retrieve("q", k=3)

    assert [r.chunk.chunk_id for r in results] == ["d", "c", "b"]


def test_mmr_ignores_cross_encoder_score_unless_every_candidate_has_one() -> None:
    pool = [
        _mk("a", "s1", 0.9, "text a", vector=[1.0, 0.0]),
        _mk("b", "s2", 0.6, "text b", vector=[0.0, 1.0]),
        _mk("c", "s3", 0.3, "text c", vector=[-1.0, 0.0]),
    ]
    pool[0].chunk.meta.extra["cross_encoder_score"] = 0.1  # only ONE candidate has it

    class _Fixed:
        def retrieve(self, query, *, k=10, filters=None):
            return pool[:k]

    # fetch_multiplier=2 -> fetch_k=4, so all 3 pool items are pulled in and
    # len(pool)=3 > k=2, ensuring the real scoring path runs (not the
    # len(pool)<=k no-op passthrough).
    retriever = MMRReranker(_Fixed(), lambda_=1.0, per_source_cap=None, fetch_multiplier=2)
    results = retriever.retrieve("q", k=2)

    # Falls back to .score entirely (a=0.9 > b=0.6 > c=0.3), not a mixed comparison.
    assert [r.chunk.chunk_id for r in results] == ["a", "b"]
