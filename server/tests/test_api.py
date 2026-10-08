"""Reference server API tests — FastAPI's TestClient, no network. The
Corpus dependency is overridden entirely (ragcore's own FakeEmbedder/
InMemoryStore test doubles), never via RAGCORE_DB_PATH, so these tests
need no real Voyage/LanceDB credentials.

`/ingest` is async as of ADR 0014 - every test that ingests uses the
`ingest_sync` fixture (server/tests/conftest.py), which drives the
real enqueue -> worker -> GET /jobs/{id} round trip and hands back the
final response, so assertions stay as close to the old synchronous
shape as the contract change allows (`resp.json()["result"][...]`
instead of `resp.json()[...]`).
"""

from __future__ import annotations

import pytest
from app.corpus_provider import get_corpus
from app.main import app
from fakes import FakeEmbedder, FakeGenerator, InMemoryStore
from fastapi.testclient import TestClient

from ragcore.corpus import Corpus


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(InMemoryStore(), FakeEmbedder())


@pytest.fixture
def client(corpus: Corpus) -> TestClient:
    app.dependency_overrides[get_corpus] = lambda: corpus
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def corpus_with_generator() -> Corpus:
    return Corpus(InMemoryStore(), FakeEmbedder(), generator=FakeGenerator())


@pytest.fixture
def client_with_generator(corpus_with_generator: Corpus) -> TestClient:
    app.dependency_overrides[get_corpus] = lambda: corpus_with_generator
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ingest_then_sources_then_retrieve(client: TestClient, corpus: Corpus, ingest_sync) -> None:
    content = b"# Chapter 1\n\nSome content about chunking and embeddings.\n"
    resp = ingest_sync(client, "book.md", content, corpus)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "finished"
    report = body["result"]
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


def test_ask_with_a_generator_returns_an_answer(
    client_with_generator: TestClient, corpus_with_generator: Corpus, ingest_sync
) -> None:
    content = b"# Chapter 1\n\nSome content about chunking.\n"
    ingest_sync(client_with_generator, "book.md", content, corpus_with_generator)

    resp = client_with_generator.post("/ask", json={"query": "chunking"})

    assert resp.status_code == 200
    body = resp.json()
    assert "answer to" in body["text"]


def test_delete_source_removes_its_chunks(client: TestClient, corpus: Corpus, ingest_sync) -> None:
    content = b"# Chapter 1\n\nSome content.\n"
    ingest_resp = ingest_sync(client, "book.md", content, corpus)
    source_id = ingest_resp.json()["result"]["source_ids"][0]

    delete_resp = client.delete(f"/sources/{source_id}")

    assert delete_resp.status_code == 200
    assert delete_resp.json()["chunks_deleted"] > 0
    assert client.get("/sources").json() == []


def test_unsupported_file_extension_fails_the_job_not_the_enqueue(
    client: TestClient, corpus: Corpus, ingest_sync
) -> None:
    """Enqueuing always succeeds (the upload itself doesn't know the
    extension is unsupported) - the failure now surfaces as the job's
    own status, not a synchronous 400, which is the one real
    behavioral consequence of ADR 0014's async /ingest contract this
    test exists to pin down."""
    resp = ingest_sync(client, "book.xyz", b"content", corpus)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["error"]


def test_get_job_status_for_an_unknown_job_id_returns_404(
    client: TestClient, _async_ingest_infra
) -> None:
    resp = client.get("/jobs/nonexistent-job-id")
    assert resp.status_code == 404


def test_get_job_status_without_redis_configured_returns_503(client: TestClient) -> None:
    """No _async_ingest_infra fixture here deliberately - REDIS_URL is
    genuinely unset in this test, pinning the distinct failure mode
    from the test above (unreachable queue vs. a real but unknown
    job_id)."""
    resp = client.get("/jobs/nonexistent-job-id")
    assert resp.status_code == 503
