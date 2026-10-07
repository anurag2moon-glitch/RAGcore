"""InMemoryUsageMeter and the GET /usage endpoint: request counts per
tenant, never leaking another tenant's counts.
"""

from __future__ import annotations

import io

import pytest
from app.auth import InMemoryApiKeyStore, get_api_key_store
from app.corpus_provider import get_corpus
from app.main import app
from app.usage import InMemoryUsageMeter, get_usage_meter
from fakes import FakeEmbedder, InMemoryStore
from fastapi.testclient import TestClient

from ragcore.corpus import Corpus


def test_record_then_usage_for_returns_the_recorded_count() -> None:
    meter = InMemoryUsageMeter()
    meter.record("tenant-a", "GET /sources")
    meter.record("tenant-a", "GET /sources")
    meter.record("tenant-a", "POST /ingest")

    assert meter.usage_for("tenant-a") == {"GET /sources": 2, "POST /ingest": 1}


def test_usage_for_an_unknown_tenant_is_empty() -> None:
    meter = InMemoryUsageMeter()
    assert meter.usage_for("nobody") == {}


def test_usage_for_one_tenant_never_includes_anothers() -> None:
    meter = InMemoryUsageMeter()
    meter.record("tenant-a", "GET /sources")
    meter.record("tenant-b", "GET /sources")

    assert meter.usage_for("tenant-a") == {"GET /sources": 1}
    assert meter.usage_for("tenant-b") == {"GET /sources": 1}


@pytest.fixture
def metered_client():
    corpus = Corpus(InMemoryStore(), FakeEmbedder())
    store = InMemoryApiKeyStore({"key-a": "tenant-a", "key-b": "tenant-b"})
    meter = InMemoryUsageMeter()  # one shared instance across requests
    app.dependency_overrides[get_corpus] = lambda: corpus
    app.dependency_overrides[get_api_key_store] = lambda: store
    app.dependency_overrides[get_usage_meter] = lambda: meter
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_usage_without_auth_returns_400(metered_client: TestClient) -> None:
    corpus = Corpus(InMemoryStore(), FakeEmbedder())
    app.dependency_overrides[get_corpus] = lambda: corpus
    app.dependency_overrides[get_api_key_store] = lambda: None
    resp = metered_client.get("/usage")
    assert resp.status_code == 400


def test_usage_counts_requests_for_this_tenant(metered_client: TestClient) -> None:
    metered_client.get("/sources", headers={"X-API-Key": "key-a"})
    metered_client.get("/sources", headers={"X-API-Key": "key-a"})

    resp = metered_client.get("/usage", headers={"X-API-Key": "key-a"})

    assert resp.status_code == 200
    assert resp.json() == {"GET /sources": 2}


def test_usage_never_includes_another_tenants_requests(metered_client: TestClient) -> None:
    metered_client.get("/sources", headers={"X-API-Key": "key-a"})
    metered_client.get("/sources", headers={"X-API-Key": "key-b"})
    metered_client.get("/sources", headers={"X-API-Key": "key-b"})

    resp_a = metered_client.get("/usage", headers={"X-API-Key": "key-a"})
    resp_b = metered_client.get("/usage", headers={"X-API-Key": "key-b"})

    assert resp_a.json() == {"GET /sources": 1}
    assert resp_b.json() == {"GET /sources": 2}


def test_ingest_requests_are_metered_separately_from_retrieve(metered_client: TestClient) -> None:
    content = b"# Chapter 1\n\nSome content.\n"
    metered_client.post(
        "/ingest",
        files={"file": ("book.md", io.BytesIO(content), "text/markdown")},
        headers={"X-API-Key": "key-a"},
    )
    metered_client.post(
        "/retrieve", json={"query": "content", "k": 5}, headers={"X-API-Key": "key-a"}
    )

    resp = metered_client.get("/usage", headers={"X-API-Key": "key-a"})

    body = resp.json()
    assert body.get("POST /ingest") == 1
    assert body.get("POST /retrieve") == 1


def test_reading_usage_does_not_count_against_itself(metered_client: TestClient) -> None:
    metered_client.get("/usage", headers={"X-API-Key": "key-a"})
    resp = metered_client.get("/usage", headers={"X-API-Key": "key-a"})

    assert "GET /usage" not in resp.json()
