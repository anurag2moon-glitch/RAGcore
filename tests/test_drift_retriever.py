"""DriftRetriever: fuses a local and a global Retriever's results via
RRF. Pure Python doubles — no embedder, no LLM, no store.
"""

from __future__ import annotations

from ragcore.retrieval.drift import DriftRetriever
from ragcore.types import Chunk, ChunkMeta, Retrieved


def _chunk(chunk_id: str, text: str = "text") -> Chunk:
    meta = ChunkMeta(
        source_id="src", source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=0, section=None,
        page_start=None, page_end=None, chunk_index=0, context_group_id="g",
        char_start=0, char_end=len(text), token_count=1,
    )
    return Chunk(chunk_id=chunk_id, text=text, meta=meta)


class FakeRetriever:
    def __init__(self, results: list[Retrieved]) -> None:
        self._results = results
        self.calls = 0

    def retrieve(self, query, *, k=10, filters=None):
        self.calls += 1
        return self._results[:k]


def test_calls_both_local_and_global() -> None:
    local = FakeRetriever([Retrieved(chunk=_chunk("c1"), score=0.5)])
    global_ = FakeRetriever([Retrieved(chunk=_chunk("c2"), score=0.9)])
    retriever = DriftRetriever(local, global_)

    retriever.retrieve("q")

    assert local.calls == 1
    assert global_.calls == 1


def test_a_chunk_found_by_both_outranks_one_found_by_only_one() -> None:
    c1, c2 = _chunk("c1"), _chunk("c2")
    local = FakeRetriever([Retrieved(chunk=c1, score=0.5), Retrieved(chunk=c2, score=0.4)])
    global_ = FakeRetriever([Retrieved(chunk=c2, score=0.9)])
    retriever = DriftRetriever(local, global_)

    results = retriever.retrieve("q")

    assert [r.chunk.chunk_id for r in results] == ["c2", "c1"]


def test_results_are_stamped_with_drift_sources() -> None:
    c1 = _chunk("c1")
    local = FakeRetriever([Retrieved(chunk=c1, score=0.5)])
    global_ = FakeRetriever([Retrieved(chunk=c1, score=0.6)])
    retriever = DriftRetriever(local, global_)

    [result] = retriever.retrieve("q")

    assert result.chunk.meta.extra["drift_sources"] == ("local", "global")


def test_k_zero_returns_empty_without_calling_either_side() -> None:
    local = FakeRetriever([Retrieved(chunk=_chunk("c1"), score=0.5)])
    global_ = FakeRetriever([Retrieved(chunk=_chunk("c2"), score=0.5)])
    retriever = DriftRetriever(local, global_)

    assert retriever.retrieve("q", k=0) == []
    assert local.calls == 0
    assert global_.calls == 0


def test_respects_k_on_the_merged_result() -> None:
    chunks = [_chunk(f"c{i}") for i in range(5)]
    local = FakeRetriever([Retrieved(chunk=c, score=1.0 - i * 0.1) for i, c in enumerate(chunks)])
    global_ = FakeRetriever([])
    retriever = DriftRetriever(local, global_)

    results = retriever.retrieve("q", k=2)

    assert len(results) == 2
