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
from fastapi.testclient import TestClient

from ragcore.corpus import Corpus
from ragcore.generation.base import Answer
from ragcore.types import Retrieved


class _FakeEmbedder:
    dim = 8
    model = "fake-embedder"

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.dim

    def embed_groups(self, groups, *, input_type="document"):
        return [[[0.1] * self.dim for _ in g] for g in groups]


class _InMemoryStore:
    dim = 8

    def __init__(self) -> None:
        self._chunks: dict[str, object] = {}

    def upsert(self, chunks) -> None:
        for c in chunks:
            self._chunks[c.chunk_id] = c

    def query(self, vector, k, filters=None):
        chunks = list(self._chunks.values())[:k]
        return [Retrieved(chunk=c, score=0.9) for c in chunks]

    def get(self, chunk_ids):
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def get_by_source(self, source_id):
        return [c for c in self._chunks.values() if c.meta.source_id == source_id]

    def delete(self, *, source_id=None, chunk_ids=None):
        if source_id is not None:
            to_remove = [cid for cid, c in self._chunks.items() if c.meta.source_id == source_id]
        else:
            to_remove = list(chunk_ids)
        for cid in to_remove:
            del self._chunks[cid]
        return len(to_remove)

    def list_sources(self):
        from ragcore.types import SourceInfo

        by_source: dict[str, list] = {}
        for c in self._chunks.values():
            by_source.setdefault(c.meta.source_id, []).append(c)
        return [
            SourceInfo(
                source_id=sid,
                path=chunks[0].meta.source_path,
                title=chunks[0].meta.source_title,
                source_type=chunks[0].meta.source_type,
                chunk_count=len(chunks),
            )
            for sid, chunks in by_source.items()
        ]

    def count(self) -> int:
        return len(self._chunks)


class _FakeGenerator:
    model = "fake-generator"

    def generate(self, query, context) -> Answer:
        return Answer(text=f"answer to {query!r}", citations=list(context))


@pytest.fixture
def client() -> TestClient:
    corpus = Corpus(_InMemoryStore(), _FakeEmbedder())
    app.dependency_overrides[get_corpus] = lambda: corpus
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_with_generator() -> TestClient:
    corpus = Corpus(_InMemoryStore(), _FakeEmbedder(), generator=_FakeGenerator())
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
