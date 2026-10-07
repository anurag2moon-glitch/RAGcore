"""InMemoryRateLimiter (token bucket semantics) and enforce_rate_limit
wired into the real app — both unit-level and integration-level, since
the unit test alone wouldn't catch a wiring mistake (e.g. applying the
dependency to the wrong routes, or not exempting /health).
"""

from __future__ import annotations

import time

import pytest
from app.auth import InMemoryApiKeyStore, get_api_key_store
from app.corpus_provider import get_corpus
from app.main import app
from app.rate_limit import InMemoryRateLimiter, get_rate_limiter
from fakes import FakeEmbedder, InMemoryStore
from fastapi.testclient import TestClient

from ragcore.corpus import Corpus


def test_rejects_a_non_positive_rate() -> None:
    with pytest.raises(ValueError, match="positive"):
        InMemoryRateLimiter(0)


def test_allows_up_to_the_configured_burst_capacity() -> None:
    limiter = InMemoryRateLimiter(requests_per_minute=3)
    assert limiter.allow("k") is True
    assert limiter.allow("k") is True
    assert limiter.allow("k") is True
    assert limiter.allow("k") is False  # 4th immediate request exceeds the burst


def test_refills_over_time() -> None:
    limiter = InMemoryRateLimiter(requests_per_minute=60)  # 1/sec
    assert limiter.allow("k") is True
    assert limiter.allow("k") is True
    time.sleep(1.05)
    assert limiter.allow("k") is True  # one token refilled


def test_different_keys_have_independent_buckets() -> None:
    limiter = InMemoryRateLimiter(requests_per_minute=1)
    assert limiter.allow("tenant-a") is True
    assert limiter.allow("tenant-a") is False
    assert limiter.allow("tenant-b") is True  # unaffected by tenant-a's bucket


@pytest.fixture
def rate_limited_client():
    corpus = Corpus(InMemoryStore(), FakeEmbedder())
    limiter = InMemoryRateLimiter(requests_per_minute=1)  # one shared instance across requests
    app.dependency_overrides[get_corpus] = lambda: corpus
    app.dependency_overrides[get_rate_limiter] = lambda: limiter
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health_is_exempt_from_rate_limiting(rate_limited_client: TestClient) -> None:
    for _ in range(5):
        assert rate_limited_client.get("/health").status_code == 200


def test_exceeding_the_rate_limit_returns_429(rate_limited_client: TestClient) -> None:
    first = rate_limited_client.get("/sources")
    second = rate_limited_client.get("/sources")

    assert first.status_code == 200
    assert second.status_code == 429


@pytest.fixture
def rate_limited_multi_tenant_client():
    corpus = Corpus(InMemoryStore(), FakeEmbedder())
    store = InMemoryApiKeyStore({"key-a": "tenant-a", "key-b": "tenant-b"})
    limiter = InMemoryRateLimiter(requests_per_minute=1)  # one shared instance across requests
    app.dependency_overrides[get_corpus] = lambda: corpus
    app.dependency_overrides[get_api_key_store] = lambda: store
    app.dependency_overrides[get_rate_limiter] = lambda: limiter
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_rate_limit_is_scoped_per_tenant_not_global(
    rate_limited_multi_tenant_client: TestClient,
) -> None:
    resp_a_1 = rate_limited_multi_tenant_client.get(
        "/sources", headers={"X-API-Key": "key-a"}
    )
    resp_a_2 = rate_limited_multi_tenant_client.get(
        "/sources", headers={"X-API-Key": "key-a"}
    )
    resp_b_1 = rate_limited_multi_tenant_client.get(
        "/sources", headers={"X-API-Key": "key-b"}
    )

    assert resp_a_1.status_code == 200
    assert resp_a_2.status_code == 429  # tenant-a's own bucket is exhausted
    assert resp_b_1.status_code == 200  # tenant-b has its own, untouched bucket
