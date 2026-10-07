"""Per-tenant quotas: enforce_quota's boundary (allowed exactly up to
the limit, rejected after), isolation between tenants, and that quota
and /usage can never disagree since both read the same
InMemoryUsageMeter. parse_int_quotas's own parsing edge cases are
covered once, in test_env_parsing.py, not duplicated here.
"""

from __future__ import annotations

import pytest
from app.auth import InMemoryApiKeyStore, get_api_key_store
from app.corpus_provider import get_corpus
from app.main import app
from app.quota import get_tenant_quotas
from app.usage import InMemoryUsageMeter, get_usage_meter
from fakes import FakeEmbedder, InMemoryStore
from fastapi.testclient import TestClient

from ragcore.corpus import Corpus


@pytest.fixture
def quota_client():
    corpus = Corpus(InMemoryStore(), FakeEmbedder())
    store = InMemoryApiKeyStore({"key-a": "tenant-a", "key-b": "tenant-b"})
    meter = InMemoryUsageMeter()  # one shared instance across requests
    app.dependency_overrides[get_corpus] = lambda: corpus
    app.dependency_overrides[get_api_key_store] = lambda: store
    app.dependency_overrides[get_usage_meter] = lambda: meter
    app.dependency_overrides[get_tenant_quotas] = lambda: {"tenant-a": 2}
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_requests_up_to_the_quota_are_allowed(quota_client: TestClient) -> None:
    first = quota_client.get("/sources", headers={"X-API-Key": "key-a"})
    second = quota_client.get("/sources", headers={"X-API-Key": "key-a"})

    assert first.status_code == 200
    assert second.status_code == 200  # brings tenant-a to exactly the limit (2)


def test_the_request_exceeding_the_quota_returns_402(quota_client: TestClient) -> None:
    quota_client.get("/sources", headers={"X-API-Key": "key-a"})
    quota_client.get("/sources", headers={"X-API-Key": "key-a"})

    third = quota_client.get("/sources", headers={"X-API-Key": "key-a"})

    assert third.status_code == 402


def test_a_tenant_with_no_configured_quota_is_unlimited(quota_client: TestClient) -> None:
    for _ in range(5):
        resp = quota_client.get("/sources", headers={"X-API-Key": "key-b"})
        assert resp.status_code == 200


def test_one_tenants_quota_does_not_affect_another(quota_client: TestClient) -> None:
    quota_client.get("/sources", headers={"X-API-Key": "key-a"})
    quota_client.get("/sources", headers={"X-API-Key": "key-a"})
    quota_client.get("/sources", headers={"X-API-Key": "key-a"})  # tenant-a now over quota

    resp_b = quota_client.get("/sources", headers={"X-API-Key": "key-b"})

    assert resp_b.status_code == 200


def test_a_quota_rejected_request_is_not_counted_against_usage(quota_client: TestClient) -> None:
    quota_client.get("/sources", headers={"X-API-Key": "key-a"})
    quota_client.get("/sources", headers={"X-API-Key": "key-a"})
    quota_client.get("/sources", headers={"X-API-Key": "key-a"})  # rejected, 402

    usage_resp = quota_client.get("/usage", headers={"X-API-Key": "key-a"})

    assert usage_resp.json() == {"GET /sources": 2}
