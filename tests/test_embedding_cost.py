"""UsageTracker unit tests, plus wiring checks for both Embedder
implementations (Phase 2 gap list item 9: cost tracking)."""

from __future__ import annotations

from typing import Any

import pytest

from ragcore.embedding.cost import UsageTracker

pytest.importorskip("voyageai")
pytest.importorskip("openai")

from ragcore.embedding.openai import OpenAIEmbedder
from ragcore.embedding.voyage import VoyageContextEmbedder
from ragcore.errors import EmbeddingError


def test_usage_tracker_without_a_price_has_no_cost_estimate() -> None:
    usage = UsageTracker()
    usage.record_success(1000)
    assert usage.total_requests == 1
    assert usage.total_tokens == 1000
    assert usage.estimated_cost_usd is None  # never a guessed number


def test_usage_tracker_with_a_price_computes_the_estimate() -> None:
    usage = UsageTracker(price_per_million_tokens_usd=2.0)
    usage.record_success(500_000)
    usage.record_success(500_000)
    assert usage.total_tokens == 1_000_000
    assert usage.total_requests == 2
    assert usage.estimated_cost_usd == 2.0


def test_usage_tracker_records_failures_separately() -> None:
    usage = UsageTracker(price_per_million_tokens_usd=1.0)
    usage.record_failure()
    usage.record_failure()
    assert usage.failed_requests == 2
    assert usage.total_tokens == 0
    assert usage.estimated_cost_usd == 0.0


# -- OpenAIEmbedder wiring -------------------------------------------------


class _StubItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _StubResponse:
    def __init__(self, data: list[_StubItem]) -> None:
        self.data = data


class _OpenAIStubClient:
    def __init__(self, fail_times: int = 0) -> None:
        self._fail_times = fail_times
        self._calls = 0

        class _Embeddings:
            def __init__(self, outer: _OpenAIStubClient) -> None:
                self._outer = outer

            def create(self, *, input: list[str], **kwargs: Any) -> _StubResponse:
                self._outer._calls += 1
                if self._outer._calls <= self._outer._fail_times:
                    raise RuntimeError("simulated transient failure")
                return _StubResponse([_StubItem([1.0, 2.0]) for _ in input])

        self.embeddings = _Embeddings(self)


def test_openai_embedder_default_price_for_a_known_model() -> None:
    embedder = OpenAIEmbedder(client=_OpenAIStubClient(), model="text-embedding-3-small", dim=2)
    assert embedder.usage.price_per_million_tokens_usd == 0.02


def test_openai_embedder_unknown_model_has_no_default_price() -> None:
    embedder = OpenAIEmbedder(client=_OpenAIStubClient(), model="some-future-model", dim=2)
    assert embedder.usage.price_per_million_tokens_usd is None


def test_openai_embedder_explicit_price_overrides_the_default() -> None:
    embedder = OpenAIEmbedder(
        client=_OpenAIStubClient(),
        model="text-embedding-3-small",
        dim=2,
        price_per_million_tokens_usd=99.0,
    )
    assert embedder.usage.price_per_million_tokens_usd == 99.0


def test_openai_embedder_records_usage_on_success() -> None:
    embedder = OpenAIEmbedder(client=_OpenAIStubClient(), dim=2)
    embedder.embed_groups([["hello world"], ["a second chunk"]])

    assert embedder.usage.total_requests == 1
    assert embedder.usage.total_tokens > 0
    assert embedder.usage.failed_requests == 0


def test_openai_embedder_records_failure_after_exhausted_retries() -> None:
    client = _OpenAIStubClient(fail_times=99)  # always fails
    embedder = OpenAIEmbedder(client=client, dim=2, max_retries=2)

    with pytest.raises(EmbeddingError):
        embedder.embed_groups([["hello"]])

    assert embedder.usage.total_requests == 0
    assert embedder.usage.total_tokens == 0
    assert embedder.usage.failed_requests == 1


# -- VoyageContextEmbedder wiring ------------------------------------------


class _VoyageStubResponse:
    def __init__(self, results: list[Any]) -> None:
        self.results = results


class _VoyageStubResult:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


class _VoyageStubClient:
    def __init__(self, fail_times: int = 0) -> None:
        self._fail_times = fail_times
        self._calls = 0

    def contextualized_embed(self, *, inputs, **kwargs):
        self._calls += 1
        if self._calls <= self._fail_times:
            raise RuntimeError("simulated transient failure")
        return _VoyageStubResponse(
            [_VoyageStubResult([[1.0, 2.0] for _ in group]) for group in inputs]
        )


def test_voyage_embedder_has_no_default_price() -> None:
    embedder = VoyageContextEmbedder(client=_VoyageStubClient(), output_dimension=2)
    assert embedder.usage.price_per_million_tokens_usd is None
    assert embedder.usage.estimated_cost_usd is None


def test_voyage_embedder_explicit_price_is_used() -> None:
    embedder = VoyageContextEmbedder(
        client=_VoyageStubClient(), output_dimension=2, price_per_million_tokens_usd=0.18
    )
    embedder.embed_groups([["some text"]])
    assert embedder.usage.price_per_million_tokens_usd == 0.18
    assert embedder.usage.estimated_cost_usd is not None
    assert embedder.usage.estimated_cost_usd > 0


def test_voyage_embedder_records_usage_on_success() -> None:
    embedder = VoyageContextEmbedder(client=_VoyageStubClient(), output_dimension=2)
    embedder.embed_groups([["hello world"], ["a second chunk"]])

    assert embedder.usage.total_requests == 1
    assert embedder.usage.total_tokens > 0
    assert embedder.usage.failed_requests == 0


def test_voyage_embedder_records_failure_after_exhausted_retries() -> None:
    client = _VoyageStubClient(fail_times=99)
    embedder = VoyageContextEmbedder(client=client, output_dimension=2, max_retries=2)

    with pytest.raises(EmbeddingError):
        embedder.embed_groups([["hello"]])

    assert embedder.usage.total_requests == 0
    assert embedder.usage.failed_requests == 1
