"""RoutedRetriever: dispatches to named Retrievers per router.route()
decision, fuses multi-mode results via RRF. Pure Python doubles — no
embedder, no LLM, no store.
"""

from __future__ import annotations

import pytest

from ragcore.retrieval.routing import RoutedRetriever
from ragcore.types import Chunk, ChunkMeta, Retrieved


def _chunk(chunk_id: str, text: str = "text") -> Chunk:
    meta = ChunkMeta(
        source_id="src", source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=None, section=None,
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


class FakeRouter:
    def __init__(self, modes: list[str]) -> None:
        self._modes = modes
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def route(self, query, available_modes):
        self.calls.append((query, tuple(available_modes)))
        return self._modes


def test_dispatches_only_to_the_routers_chosen_mode() -> None:
    vector = FakeRetriever([Retrieved(chunk=_chunk("c1"), score=0.5)])
    keyword = FakeRetriever([Retrieved(chunk=_chunk("c2"), score=0.9)])
    retriever = RoutedRetriever({"vector": vector, "keyword": keyword}, FakeRouter(["vector"]))

    results = retriever.retrieve("q")

    assert vector.calls == 1
    assert keyword.calls == 0
    assert [r.chunk.chunk_id for r in results] == ["c1"]


def test_combination_mode_fuses_both_retrievers_results() -> None:
    c1, c2 = _chunk("c1"), _chunk("c2")
    vector = FakeRetriever([Retrieved(chunk=c1, score=0.5), Retrieved(chunk=c2, score=0.4)])
    keyword = FakeRetriever([Retrieved(chunk=c2, score=0.9)])
    retriever = RoutedRetriever(
        {"vector": vector, "keyword": keyword}, FakeRouter(["vector", "keyword"])
    )

    results = retriever.retrieve("q")

    # c2 found by both modes outranks c1 (found by only one), per RRF.
    assert [r.chunk.chunk_id for r in results] == ["c2", "c1"]


def test_results_are_stamped_with_contributing_modes() -> None:
    c1 = _chunk("c1")
    vector = FakeRetriever([Retrieved(chunk=c1, score=0.5)])
    keyword = FakeRetriever([Retrieved(chunk=c1, score=0.6)])
    retriever = RoutedRetriever(
        {"vector": vector, "keyword": keyword}, FakeRouter(["vector", "keyword"])
    )

    [result] = retriever.retrieve("q")

    assert result.chunk.meta.extra["routed_modes"] == ("vector", "keyword")


def test_falls_back_to_default_modes_when_router_returns_nothing_recognized() -> None:
    vector = FakeRetriever([Retrieved(chunk=_chunk("c1"), score=0.5)])
    retriever = RoutedRetriever({"vector": vector}, FakeRouter(["made_up_mode"]))

    results = retriever.retrieve("q")

    assert vector.calls == 1
    assert [r.chunk.chunk_id for r in results] == ["c1"]


def test_falls_back_to_explicit_default_modes_not_every_mode() -> None:
    vector = FakeRetriever([Retrieved(chunk=_chunk("c1"), score=0.5)])
    keyword = FakeRetriever([Retrieved(chunk=_chunk("c2"), score=0.5)])
    retriever = RoutedRetriever(
        {"vector": vector, "keyword": keyword},
        FakeRouter([]),
        default_modes=["keyword"],
    )

    retriever.retrieve("q")

    assert vector.calls == 0
    assert keyword.calls == 1


def test_last_route_records_the_modes_actually_used() -> None:
    vector = FakeRetriever([Retrieved(chunk=_chunk("c1"), score=0.5)])
    retriever = RoutedRetriever({"vector": vector}, FakeRouter(["vector"]))

    retriever.retrieve("q")

    assert retriever.last_route == ("vector",)


def test_router_is_given_the_available_mode_names() -> None:
    vector = FakeRetriever([])
    keyword = FakeRetriever([])
    router = FakeRouter(["vector"])
    retriever = RoutedRetriever({"vector": vector, "keyword": keyword}, router)

    retriever.retrieve("the query")

    assert router.calls == [("the query", ("vector", "keyword"))]


def test_k_zero_returns_empty_without_calling_the_router() -> None:
    vector = FakeRetriever([Retrieved(chunk=_chunk("c1"), score=0.5)])
    router = FakeRouter(["vector"])
    retriever = RoutedRetriever({"vector": vector}, router)

    assert retriever.retrieve("q", k=0) == []
    assert router.calls == []
    assert vector.calls == 0


def test_requires_at_least_one_retriever() -> None:
    with pytest.raises(ValueError, match="at least one"):
        RoutedRetriever({}, FakeRouter([]))
