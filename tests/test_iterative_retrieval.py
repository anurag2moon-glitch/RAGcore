"""IterativeRetriever: hop cap, sufficiency-driven early stop, hop
provenance stamping, and the earliest-hop/max-score merge rule. No network —
a deterministic fake SufficiencyJudge stands in for ClaudeSufficiencyJudge.
"""

from __future__ import annotations

from ragcore.retrieval.iterative import IterativeRetriever, Sufficiency
from ragcore.types import Chunk, ChunkMeta, Retrieved


def _mk(chunk_id: str, score: float, text: str = "text") -> Retrieved:
    meta = ChunkMeta(
        source_id="s", source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=None, section=None,
        page_start=None, page_end=None, chunk_index=0, context_group_id="g",
        char_start=0, char_end=len(text), token_count=1,
    )
    return Retrieved(chunk=Chunk(chunk_id=chunk_id, text=text, meta=meta), score=score)


class _ScriptedRetriever:
    """Returns a different fixed result set per query, recording calls."""

    def __init__(self, results_by_query: dict[str, list[Retrieved]]) -> None:
        self._results_by_query = results_by_query
        self.calls: list[str] = []

    def retrieve(self, query, *, k=10, filters=None):
        self.calls.append(query)
        return self._results_by_query.get(query, [])[:k]


class _ScriptedJudge:
    model = "scripted"

    def __init__(self, decisions: list[Sufficiency]) -> None:
        self._decisions = decisions
        self.calls: list[tuple[str, int]] = []

    def assess(self, original_query, hops_so_far, context):
        self.calls.append((original_query, len(hops_so_far)))
        return self._decisions[len(hops_so_far)]


def test_stops_after_first_hop_when_judge_says_sufficient() -> None:
    inner = _ScriptedRetriever({"q": [_mk("a", 0.9), _mk("b", 0.8)]})
    judge = _ScriptedJudge([Sufficiency(sufficient=True, next_query=None, reasoning="enough")])
    retriever = IterativeRetriever(inner, judge, max_hops=3)

    results = retriever.retrieve("q", k=5)

    assert [r.chunk.chunk_id for r in results] == ["a", "b"]
    assert inner.calls == ["q"]
    assert len(retriever.last_hops) == 1
    assert retriever.last_hops[0].sufficient is True


def test_continues_to_a_refined_query_when_insufficient() -> None:
    inner = _ScriptedRetriever(
        {"q": [_mk("a", 0.9)], "refined": [_mk("b", 0.7)]}
    )
    judge = _ScriptedJudge(
        [
            Sufficiency(sufficient=False, next_query="refined", reasoning="need more"),
            Sufficiency(sufficient=True, next_query=None, reasoning="now enough"),
        ]
    )
    retriever = IterativeRetriever(inner, judge, max_hops=3)

    results = retriever.retrieve("q", k=5)

    assert inner.calls == ["q", "refined"]
    ids = {r.chunk.chunk_id for r in results}
    assert ids == {"a", "b"}
    assert len(retriever.last_hops) == 2


def test_stops_at_max_hops_even_if_judge_never_says_sufficient() -> None:
    inner = _ScriptedRetriever(
        {"q": [_mk("a", 0.9)], "q2": [_mk("b", 0.8)], "q3": [_mk("c", 0.7)]}
    )
    # Always insufficient, always proposes the next scripted query.
    judge = _ScriptedJudge(
        [
            Sufficiency(sufficient=False, next_query="q2", reasoning="r1"),
            Sufficiency(sufficient=False, next_query="q3", reasoning="r2"),
            Sufficiency(sufficient=False, next_query="q4", reasoning="r3"),
        ]
    )
    retriever = IterativeRetriever(inner, judge, max_hops=3)

    retriever.retrieve("q", k=5)

    assert inner.calls == ["q", "q2", "q3"]  # capped at 3 hops, never calls with "q4"
    assert len(retriever.last_hops) == 3
    assert retriever.last_hops[-1].sufficient is False


def test_stops_early_when_judge_proposes_no_next_query_despite_insufficient() -> None:
    inner = _ScriptedRetriever({"q": [_mk("a", 0.9)]})
    judge = _ScriptedJudge([Sufficiency(sufficient=False, next_query=None, reasoning="stuck")])
    retriever = IterativeRetriever(inner, judge, max_hops=3)

    retriever.retrieve("q", k=5)

    assert inner.calls == ["q"]


def test_hop_provenance_is_stamped_on_every_returned_chunk() -> None:
    inner = _ScriptedRetriever({"q": [_mk("a", 0.9)], "q2": [_mk("b", 0.5)]})
    judge = _ScriptedJudge(
        [
            Sufficiency(sufficient=False, next_query="q2", reasoning="r1"),
            Sufficiency(sufficient=True, next_query=None, reasoning="r2"),
        ]
    )
    retriever = IterativeRetriever(inner, judge, max_hops=3)

    results = retriever.retrieve("q", k=5)

    by_id = {r.chunk.chunk_id: r for r in results}
    assert by_id["a"].chunk.meta.extra["hop_index"] == 0
    assert by_id["a"].chunk.meta.extra["hop_query"] == "q"
    assert by_id["b"].chunk.meta.extra["hop_index"] == 1
    assert by_id["b"].chunk.meta.extra["hop_query"] == "q2"


def test_a_chunk_rediscovered_in_a_later_hop_keeps_its_earliest_hop_stamp_but_best_score() -> None:
    inner = _ScriptedRetriever({"q": [_mk("a", 0.5)], "q2": [_mk("a", 0.9)]})
    judge = _ScriptedJudge(
        [
            Sufficiency(sufficient=False, next_query="q2", reasoning="r1"),
            Sufficiency(sufficient=True, next_query=None, reasoning="r2"),
        ]
    )
    retriever = IterativeRetriever(inner, judge, max_hops=3)

    results = retriever.retrieve("q", k=5)

    assert len(results) == 1
    assert results[0].chunk.meta.extra["hop_index"] == 0  # earliest hop kept
    assert results[0].chunk.meta.extra["hop_query"] == "q"
    assert results[0].score == 0.9  # but the higher score wins


def test_k_le_0_returns_empty_and_clears_last_hops() -> None:
    inner = _ScriptedRetriever({"q": [_mk("a", 0.9)]})
    judge = _ScriptedJudge([Sufficiency(sufficient=True, next_query=None, reasoning="x")])
    retriever = IterativeRetriever(inner, judge, max_hops=3)

    assert retriever.retrieve("q", k=0) == []
    assert retriever.last_hops == ()
    assert inner.calls == []


def test_results_are_capped_at_k_across_all_hops_combined() -> None:
    inner = _ScriptedRetriever(
        {"q": [_mk("a", 0.9), _mk("b", 0.8)], "q2": [_mk("c", 0.95), _mk("d", 0.1)]}
    )
    judge = _ScriptedJudge(
        [
            Sufficiency(sufficient=False, next_query="q2", reasoning="r1"),
            Sufficiency(sufficient=True, next_query=None, reasoning="r2"),
        ]
    )
    retriever = IterativeRetriever(inner, judge, max_hops=3)

    results = retriever.retrieve("q", k=2)

    assert len(results) == 2
    assert [r.chunk.chunk_id for r in results] == ["c", "a"]  # top 2 by score across both hops
