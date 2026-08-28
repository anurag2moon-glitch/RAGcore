"""Unit tests for VoyageContextEmbedder's batching, against a stub client
(no network, no real API key). Covers the token-budget gap found ingesting
a real book: an account-tier token-per-minute cap can reject a request
that's well within the structural (input/chunk count) limits.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("voyageai")

from ragcore.embedding.voyage import VoyageContextEmbedder
from ragcore.errors import EmbeddingError


class _StubResponse:
    def __init__(self, results: list[Any]) -> None:
        self.results = results


class _StubResult:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


class _StubClient:
    def __init__(self) -> None:
        self.calls: list[list[list[str]]] = []

    def contextualized_embed(self, *, inputs, **kwargs):
        self.calls.append(inputs)
        # one fixed-length fake vector per chunk, per group, in order
        return _StubResponse([_StubResult([[1.0, 2.0] for _ in group]) for group in inputs])


def _embedder(**kwargs) -> tuple[VoyageContextEmbedder, _StubClient]:
    client = _StubClient()
    embedder = VoyageContextEmbedder(client=client, output_dimension=2, **kwargs)
    return embedder, client


def test_batches_split_on_token_budget_not_just_chunk_count() -> None:
    # Each group is one short sentence (~a few tokens). A tiny token budget
    # forces multiple requests even though chunk/input counts are trivial.
    embedder, client = _embedder(max_tokens_per_request=15, min_request_interval=0.0)
    groups = [["one two three four five"] for _ in range(10)]

    vectors = embedder.embed_groups(groups)

    assert len(vectors) == 10
    assert all(len(g) == 1 for g in vectors)
    assert len(client.calls) > 1, "expected the tiny token budget to force multiple requests"
    assert sum(len(call) for call in client.calls) == 10


def test_results_reassembled_in_original_order_across_batches() -> None:
    embedder, client = _embedder(max_tokens_per_request=10)
    groups = [[f"chunk number {i}"] for i in range(6)]

    vectors = embedder.embed_groups(groups)

    assert len(vectors) == 6
    assert len(client.calls) > 1


def test_oversized_single_group_raises_instead_of_silently_splitting() -> None:
    embedder, _ = _embedder(max_tokens_per_request=5)
    huge_group = ["this sentence alone already has more than five tokens in it"]

    with pytest.raises(EmbeddingError, match="max_tokens_per_request"):
        embedder.embed_groups([huge_group])


def test_min_request_interval_paces_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    # max_tokens_per_request=1 forces one group per request regardless of
    # exact tokenization, since any two non-empty groups together exceed it.
    embedder, client = _embedder(max_tokens_per_request=1, min_request_interval=0.05)
    groups = [["a"], ["b"], ["c"]]

    sleeps: list[float] = []
    monkeypatch.setattr("ragcore.embedding.voyage.time.sleep", lambda s: sleeps.append(s))

    embedder.embed_groups(groups)

    assert len(client.calls) == 3
    assert len(sleeps) == 2  # no wait before the first request, then one wait per subsequent request
    assert all(s > 0 for s in sleeps)
