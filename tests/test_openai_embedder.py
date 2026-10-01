"""Unit tests for OpenAIEmbedder's batching/flattening, against a stub
client (no network, no real API key) — same pattern as
test_voyage_batching.py.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("openai")

from ragcore.embedding.openai import OpenAIEmbedder
from ragcore.errors import EmbeddingError


class _StubItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _StubResponse:
    def __init__(self, data: list[_StubItem]) -> None:
        self.data = data


class _StubClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

        class _Embeddings:
            def __init__(self, outer: _StubClient) -> None:
                self._outer = outer

            def create(self, *, input: list[str], **kwargs: Any) -> _StubResponse:
                self._outer.calls.append(list(input))
                return _StubResponse([_StubItem([1.0, 2.0]) for _ in input])

        self.embeddings = _Embeddings(self)


def _embedder(**kwargs: Any) -> tuple[OpenAIEmbedder, _StubClient]:
    client = _StubClient()
    embedder = OpenAIEmbedder(client=client, dim=2, **kwargs)
    return embedder, client


def test_flattens_groups_into_independent_inputs_and_reassembles() -> None:
    embedder, client = _embedder()
    groups = [["a1", "a2", "a3"], ["b1"], ["c1", "c2"]]

    vectors = embedder.embed_groups(groups)

    assert [len(g) for g in vectors] == [3, 1, 2]
    # exactly one request, since nothing here exceeds any limit
    assert client.calls == [["a1", "a2", "a3", "b1", "c1", "c2"]]


def test_batches_split_on_token_budget() -> None:
    embedder, client = _embedder(max_tokens_per_request=15)
    groups = [["one two three four five"] for _ in range(10)]

    vectors = embedder.embed_groups(groups)

    assert len(vectors) == 10
    assert all(len(g) == 1 for g in vectors)
    assert len(client.calls) > 1
    assert sum(len(call) for call in client.calls) == 10


def test_results_reassembled_in_original_order_across_batches() -> None:
    embedder, client = _embedder(max_tokens_per_request=10)
    groups = [[f"chunk number {i}"] for i in range(6)]

    vectors = embedder.embed_groups(groups)

    assert len(vectors) == 6
    assert len(client.calls) > 1


def test_a_single_oversized_input_raises() -> None:
    embedder, _ = _embedder(max_tokens_per_request=5)
    huge_group = ["this sentence alone already has more than five tokens in it"]

    with pytest.raises(EmbeddingError, match="max_tokens_per_request"):
        embedder.embed_groups([huge_group])


def test_empty_groups_list_returns_empty_without_a_request() -> None:
    embedder, client = _embedder()
    assert embedder.embed_groups([]) == []
    assert client.calls == []


def test_group_with_no_texts_returns_an_empty_vector_list_for_it() -> None:
    embedder, _client = _embedder()
    vectors = embedder.embed_groups([[], ["only text"]])
    assert vectors == [[], [[1.0, 2.0]]]


def test_input_type_is_accepted_but_has_no_request_effect() -> None:
    embedder, client = _embedder()
    embedder.embed_groups([["text"]], input_type="query")
    embedder.embed_groups([["text"]], input_type="document")
    assert len(client.calls) == 2  # both calls succeeded; no error from either input_type


def test_embed_query_returns_a_single_vector() -> None:
    embedder, client = _embedder()
    vector = embedder.embed_query("a single query")
    assert vector == [1.0, 2.0]
    assert client.calls == [["a single query"]]


def test_min_request_interval_paces_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder, client = _embedder(max_tokens_per_request=1, min_request_interval=0.05)
    groups = [["a"], ["b"], ["c"]]

    sleeps: list[float] = []
    monkeypatch.setattr("ragcore.embedding.openai.time.sleep", lambda s: sleeps.append(s))

    embedder.embed_groups(groups)

    assert len(client.calls) == 3
    assert len(sleeps) == 2
    assert all(s > 0 for s in sleeps)


def test_missing_extra_raises_without_a_client() -> None:
    import ragcore.embedding.openai as openai_module

    original = openai_module.openai
    openai_module.openai = None
    try:
        with pytest.raises(EmbeddingError, match=r"ragcore\[openai\]"):
            OpenAIEmbedder()
    finally:
        openai_module.openai = original
