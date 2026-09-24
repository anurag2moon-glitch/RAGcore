"""VoyageReranker: maps RerankingObject.results into RerankHit correctly,
and retries broadly (matching VoyageContextEmbedder's own convention, not
the Claude components' retryable/non-retryable split) before wrapping in
GenerationError. No API calls: a stub client stands in for voyageai.Client.
"""

from __future__ import annotations

from collections import namedtuple

import pytest

pytest.importorskip("voyageai")

from ragcore.errors import GenerationError
from ragcore.reranking.voyage import VoyageReranker

_RerankingResult = namedtuple("_RerankingResult", ["index", "document", "relevance_score"])


class _StubResponse:
    def __init__(self, results: list) -> None:
        self.results = results


class _StubClient:
    def __init__(self, results: list, *, fail_times: int = 0) -> None:
        self._results = results
        self._fail_times = fail_times
        self.calls: list[dict] = []

    def rerank(self, query, documents, *, model, top_k=None, truncation=True):
        self.calls.append({"query": query, "documents": list(documents), "model": model, "top_k": top_k})
        if len(self.calls) <= self._fail_times:
            raise RuntimeError("simulated transient failure")
        return _StubResponse(self._results)


def test_maps_reranking_results_into_rerank_hits() -> None:
    results = [
        _RerankingResult(index=1, document="beta", relevance_score=0.9),
        _RerankingResult(index=0, document="alpha", relevance_score=0.2),
    ]
    client = _StubClient(results)
    reranker = VoyageReranker(client=client)

    hits = reranker.rerank("query", ["alpha", "beta"])

    assert [(h.index, h.relevance_score) for h in hits] == [(1, 0.9), (0, 0.2)]
    assert client.calls[0]["model"] == "rerank-2.5"


def test_empty_documents_makes_no_api_call() -> None:
    client = _StubClient([])
    reranker = VoyageReranker(client=client)

    assert reranker.rerank("q", []) == []
    assert client.calls == []


def test_retries_broadly_then_wraps_in_generation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ragcore.reranking.voyage.time.sleep", lambda s: None)
    client = _StubClient([], fail_times=99)
    reranker = VoyageReranker(client=client, max_retries=2)

    with pytest.raises(GenerationError, match="reranking failed after 2 attempts"):
        reranker.rerank("q", ["a"])

    assert len(client.calls) == 2


def test_succeeds_after_a_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ragcore.reranking.voyage.time.sleep", lambda s: None)
    results = [_RerankingResult(index=0, document="a", relevance_score=0.5)]
    client = _StubClient(results, fail_times=1)
    reranker = VoyageReranker(client=client, max_retries=3)

    hits = reranker.rerank("q", ["a"])

    assert len(hits) == 1
    assert len(client.calls) == 2
