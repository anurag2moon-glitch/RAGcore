"""Multi-tenant HTTP routing (Phase 7 item 39, gated on auth existing
per docs/adr/0008): RAGCORE_API_KEYS unset -> legacy single-tenant
behavior (already covered by test_api.py, not duplicated here); once
an ApiKeyStore is wired in, every route requires a valid key and is
scoped through ragcore.tenancy.TenantScopedCorpus.
"""

from __future__ import annotations

import io

import pytest
from app.auth import InMemoryApiKeyStore, _parse_api_keys_env, get_api_key_store
from app.corpus_provider import get_corpus
from app.main import app
from fakes import FakeEmbedder, InMemoryStore
from fastapi.testclient import TestClient

from ragcore.corpus import Corpus


def _book(label: str) -> bytes:
    return f"# Chapter 1\n\n{label} content about rockets and propulsion.\n".encode()


@pytest.fixture
def auth_client() -> TestClient:
    corpus = Corpus(InMemoryStore(), FakeEmbedder())
    store = InMemoryApiKeyStore({"key-a": "tenant-a", "key-b": "tenant-b"})
    app.dependency_overrides[get_corpus] = lambda: corpus
    app.dependency_overrides[get_api_key_store] = lambda: store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_parse_api_keys_env() -> None:
    assert _parse_api_keys_env("key1:tenant-a,key2:tenant-b") == {
        "key1": "tenant-a",
        "key2": "tenant-b",
    }


def test_parse_api_keys_env_rejects_a_malformed_entry() -> None:
    with pytest.raises(RuntimeError, match="not in 'key:tenant_id' form"):
        _parse_api_keys_env("no-colon-here")


def test_missing_api_key_returns_401(auth_client: TestClient) -> None:
    resp = auth_client.get("/sources")
    assert resp.status_code == 401


def test_invalid_api_key_returns_401(auth_client: TestClient) -> None:
    resp = auth_client.get("/sources", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_valid_api_key_via_bearer_header_succeeds(auth_client: TestClient) -> None:
    resp = auth_client.get("/sources", headers={"Authorization": "Bearer key-a"})
    assert resp.status_code == 200


def test_valid_api_key_via_x_api_key_header_succeeds(auth_client: TestClient) -> None:
    resp = auth_client.get("/sources", headers={"X-API-Key": "key-a"})
    assert resp.status_code == 200


def test_ingest_tags_retrieved_chunks_with_the_resolved_tenant(auth_client: TestClient) -> None:
    auth_client.post(
        "/ingest",
        files={"file": ("book.md", io.BytesIO(_book("Tenant A")), "text/markdown")},
        headers={"X-API-Key": "key-a"},
    )

    resp = auth_client.post(
        "/retrieve", json={"query": "rockets", "k": 5}, headers={"X-API-Key": "key-a"}
    )

    results = resp.json()["results"]
    assert results
    assert all(r["tenant_id"] == "tenant-a" for r in results)


def test_two_tenants_cannot_see_each_others_sources(auth_client: TestClient) -> None:
    auth_client.post(
        "/ingest",
        files={"file": ("book_a.md", io.BytesIO(_book("Tenant A")), "text/markdown")},
        headers={"X-API-Key": "key-a"},
    )
    auth_client.post(
        "/ingest",
        files={"file": ("book_b.md", io.BytesIO(_book("Tenant B")), "text/markdown")},
        headers={"X-API-Key": "key-b"},
    )

    sources_a = auth_client.get("/sources", headers={"X-API-Key": "key-a"}).json()
    sources_b = auth_client.get("/sources", headers={"X-API-Key": "key-b"}).json()

    assert len(sources_a) == 1
    assert len(sources_b) == 1
    assert sources_a[0]["source_id"] != sources_b[0]["source_id"]


def test_retrieve_never_returns_another_tenants_chunks(auth_client: TestClient) -> None:
    auth_client.post(
        "/ingest",
        files={"file": ("book_a.md", io.BytesIO(_book("Tenant A")), "text/markdown")},
        headers={"X-API-Key": "key-a"},
    )
    auth_client.post(
        "/ingest",
        files={"file": ("book_b.md", io.BytesIO(_book("Tenant B")), "text/markdown")},
        headers={"X-API-Key": "key-b"},
    )

    resp = auth_client.post(
        "/retrieve", json={"query": "rockets", "k": 50}, headers={"X-API-Key": "key-a"}
    )

    results = resp.json()["results"]
    assert results
    assert all(r["tenant_id"] == "tenant-a" for r in results)


def test_deleting_another_tenants_source_returns_403(auth_client: TestClient) -> None:
    ingest_resp = auth_client.post(
        "/ingest",
        files={"file": ("book_b.md", io.BytesIO(_book("Tenant B")), "text/markdown")},
        headers={"X-API-Key": "key-b"},
    )
    source_id_b = ingest_resp.json()["source_ids"][0]

    delete_resp = auth_client.delete(f"/sources/{source_id_b}", headers={"X-API-Key": "key-a"})

    assert delete_resp.status_code == 403


def test_deleting_this_tenants_own_source_succeeds(auth_client: TestClient) -> None:
    ingest_resp = auth_client.post(
        "/ingest",
        files={"file": ("book_a.md", io.BytesIO(_book("Tenant A")), "text/markdown")},
        headers={"X-API-Key": "key-a"},
    )
    source_id_a = ingest_resp.json()["source_ids"][0]

    delete_resp = auth_client.delete(f"/sources/{source_id_a}", headers={"X-API-Key": "key-a"})

    assert delete_resp.status_code == 200
    assert delete_resp.json()["chunks_deleted"] > 0
