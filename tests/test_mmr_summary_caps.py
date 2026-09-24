"""MMRReranker's per-kind summary caps: community_summary and
hierarchy_summary get independent budgets, max_summaries stays a
backward-compatible alias for community_summary's cap, and a hierarchy
summary (unlike a community summary) is NOT exempt from the per-source cap.
"""

from __future__ import annotations

from collections import Counter

from ragcore.retrieval.mmr import MMRReranker
from ragcore.types import Chunk, ChunkMeta, Retrieved


def _mk(chunk_id: str, source_id: str, score: float, *, kind: str | None = None) -> Retrieved:
    meta = ChunkMeta(
        source_id=source_id, source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=None, section=None,
        page_start=None, page_end=None, chunk_index=0, context_group_id="g",
        char_start=0, char_end=1, token_count=1,
        extra={"ragcore_kind": kind} if kind else {},
    )
    return Retrieved(chunk=Chunk(chunk_id=chunk_id, text="text", meta=meta, vector=[1.0, 0.0]), score=score)


class _Fixed:
    def __init__(self, results: list[Retrieved]) -> None:
        self._results = results

    def retrieve(self, query, *, k=10, filters=None):
        return self._results[:k]


def test_community_and_hierarchy_summaries_have_independent_caps() -> None:
    # Pool deliberately larger than k so the len(pool) <= k no-op path
    # (which bypasses every cap, by design) never triggers.
    pool = (
        [_mk(f"cs{i}", f"community:{i}", 0.9, kind="community_summary") for i in range(3)]
        + [_mk(f"hs{i}", "book1", 0.8, kind="hierarchy_summary") for i in range(3)]
        + [_mk(f"p{i}", f"book{i}", 0.5) for i in range(5)]
    )
    retriever = MMRReranker(
        _Fixed(pool), lambda_=1.0, per_source_cap=None, fetch_multiplier=2,
        summary_caps={"community_summary": 1, "hierarchy_summary": 2},
    )

    results = retriever.retrieve("q", k=10)

    kinds = Counter(r.chunk.meta.extra.get("ragcore_kind") for r in results)
    assert kinds["community_summary"] == 1
    assert kinds["hierarchy_summary"] == 2


def test_max_summaries_alias_seeds_only_the_community_summary_cap() -> None:
    pool = (
        [_mk(f"cs{i}", f"community:{i}", 0.9, kind="community_summary") for i in range(3)]
        + [_mk(f"hs{i}", "book1", 0.8, kind="hierarchy_summary") for i in range(3)]
        + [_mk(f"p{i}", f"book{i}", 0.5) for i in range(5)]
    )
    retriever = MMRReranker(
        _Fixed(pool), lambda_=1.0, per_source_cap=None, fetch_multiplier=2, max_summaries=1
    )

    results = retriever.retrieve("q", k=10)

    kinds = Counter(r.chunk.meta.extra.get("ragcore_kind") for r in results)
    assert kinds["community_summary"] == 1
    assert kinds["hierarchy_summary"] == 2  # untouched default, not affected by max_summaries


def test_hierarchy_summary_is_not_exempt_from_the_per_source_cap() -> None:
    # Two hierarchy summaries plus two plain chunks all belonging to the
    # SAME book — with per_source_cap=2, the book's total presence (summary
    # + chunk combined) must still be capped at 2, unlike a community
    # summary which would be exempt.
    # Enough distinct other-source candidates that the per-source cap never
    # needs to *relax* to reach k (relaxation is correct-but-different
    # behavior, covered by test_mmr.py's own relaxation test — this test is
    # specifically about the cap holding when it doesn't need to relax).
    pool = [
        _mk("hs0", "book1", 0.95, kind="hierarchy_summary"),
        _mk("hs1", "book1", 0.9, kind="hierarchy_summary"),
        _mk("p0", "book1", 0.85),
        _mk("p1", "book1", 0.8),
        _mk("other1", "book2", 0.2),
        _mk("other2", "book3", 0.15),
        _mk("other3", "book4", 0.1),
    ]
    retriever = MMRReranker(
        _Fixed(pool), lambda_=1.0, per_source_cap=2, fetch_multiplier=2,
        summary_caps={"hierarchy_summary": 5},  # summary-kind cap deliberately generous
    )

    results = retriever.retrieve("q", k=5)

    book1_count = sum(1 for r in results if r.chunk.meta.source_id == "book1")
    assert book1_count <= 2


def test_community_summary_stays_exempt_from_the_per_source_cap() -> None:
    pool = [
        _mk("cs0", "community:x", 0.95, kind="community_summary"),
        _mk("cs1", "community:y", 0.9, kind="community_summary"),
        _mk("p0", "book1", 0.85),
        _mk("p1", "book1", 0.8),
        _mk("p2", "book1", 0.1),
    ]
    retriever = MMRReranker(_Fixed(pool), lambda_=1.0, per_source_cap=2, fetch_multiplier=1)

    results = retriever.retrieve("q", k=5)

    ids = {r.chunk.chunk_id for r in results}
    assert {"cs0", "cs1"} <= ids  # both community summaries survive regardless of per-source cap
