"""Reference server API tests — FastAPI's TestClient, no network. The
Corpus dependency is overridden entirely (ragcore's own FakeEmbedder/
InMemoryStore test doubles), never via RAGCORE_DB_PATH, so these tests
need no real Voyage/LanceDB credentials.
"""

from __future__ import annotations

import io

import pytest
from app.corpus_provider import get_corpus
from app.main import app
from fakes import FakeEmbedder, FakeGenerator, InMemoryStore
from fastapi.testclient import TestClient

from ragcore.corpus import Corpus


@pytest.fixture
def client() -> TestClient:
    corpus = Corpus(InMemoryStore(), FakeEmbedder())
    app.dependency_overrides[get_corpus] = lambda: corpus
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_with_generator() -> TestClient:
    corpus = Corpus(InMemoryStore(), FakeEmbedder(), generator=FakeGenerator())
    app.dependency_overrides[get_corpus] = lambda: corpus
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ingest_then_sources_then_retrieve(client: TestClient) -> None:
    content = b"# Chapter 1\n\nSome content about chunking and embeddings.\n"
    resp = client.post("/ingest", files={"file": ("book.md", io.BytesIO(content), "text/markdown")})
    assert resp.status_code == 200
    report = resp.json()
    assert report["chunks_ingested"] > 0
    source_id = report["source_ids"][0]

    sources_resp = client.get("/sources")
    assert sources_resp.status_code == 200
    sources = sources_resp.json()
    assert any(s["source_id"] == source_id for s in sources)

    retrieve_resp = client.post("/retrieve", json={"query": "chunking", "k": 5})
    assert retrieve_resp.status_code == 200
    results = retrieve_resp.json()["results"]
    assert results
    assert all("chunk_id" in r for r in results)


def test_ask_without_a_generator_returns_400(client: TestClient) -> None:
    resp = client.post("/ask", json={"query": "anything"})
    assert resp.status_code == 400


def test_ask_with_a_generator_returns_an_answer(client_with_generator: TestClient) -> None:
    content = b"# Chapter 1\n\nSome content about chunking.\n"
    client_with_generator.post(
        "/ingest", files={"file": ("book.md", io.BytesIO(content), "text/markdown")}
    )

    resp = client_with_generator.post("/ask", json={"query": "chunking"})

    assert resp.status_code == 200
    body = resp.json()
    assert "answer to" in body["text"]


def test_delete_source_removes_its_chunks(client: TestClient) -> None:
    content = b"# Chapter 1\n\nSome content.\n"
    ingest_resp = client.post(
        "/ingest", files={"file": ("book.md", io.BytesIO(content), "text/markdown")}
    )
    source_id = ingest_resp.json()["source_ids"][0]

    delete_resp = client.delete(f"/sources/{source_id}")

    assert delete_resp.status_code == 200
    assert delete_resp.json()["chunks_deleted"] > 0
    assert client.get("/sources").json() == []


def test_unsupported_file_extension_returns_400(client: TestClient) -> None:
    resp = client.post(
        "/ingest", files={"file": ("book.xyz", io.BytesIO(b"content"), "application/octet-stream")}
    )
    assert resp.status_code == 400
