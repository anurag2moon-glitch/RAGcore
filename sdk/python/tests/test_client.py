"""RagcoreClient, tested against the real server/ FastAPI app object
in-process via FastAPI's TestClient — no network, no running server
process. TestClient is itself an httpx.Client subclass (a sync-compatible
bridge onto the ASGI app), which is what RagcoreClient's http_client
parameter expects; a bare httpx.Client(transport=httpx.ASGITransport(...))
does not work here — confirmed directly: ASGITransport only implements
the async transport interface (handle_async_request/aclose), and raises
AttributeError the moment a sync httpx.Client tries to use it.

server/'s own Corpus dependency is overridden with the same in-memory
test doubles server/tests/test_api.py uses, imported directly from
there rather than redefined, so the two test suites can't silently
drift out of sync with each other.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_SERVER_DIR = Path(__file__).resolve().parents[3] / "server"
sys.path.insert(0, str(_SERVER_DIR))

from app.corpus_provider import get_corpus
from app.main import app

from ragcore.corpus import Corpus
from ragcore.generation.base import Answer
from ragcore.types import Retrieved, SourceInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragcore_client import RagcoreClient, RagcoreClientError


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
        return [Retrieved(chunk=c, score=0.9) for c in list(self._chunks.values())[:k]]

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
def ragcore_client(tmp_path):
    corpus = Corpus(_InMemoryStore(), _FakeEmbedder())
    app.dependency_overrides[get_corpus] = lambda: corpus
    client = RagcoreClient(http_client=TestClient(app))
    yield client, tmp_path
    client.close()
    app.dependency_overrides.clear()


@pytest.fixture
def ragcore_client_with_generator(tmp_path):
    corpus = Corpus(_InMemoryStore(), _FakeEmbedder(), generator=_FakeGenerator())
    app.dependency_overrides[get_corpus] = lambda: corpus
    client = RagcoreClient(http_client=TestClient(app))
    yield client, tmp_path
    client.close()
    app.dependency_overrides.clear()


def test_health(ragcore_client) -> None:
    client, _tmp_path = ragcore_client
    assert client.health() is True


def test_ingest_then_sources_then_retrieve(ragcore_client) -> None:
    client, tmp_path = ragcore_client
    book = tmp_path / "book.md"
    book.write_text("# Chapter 1\n\nSome content about chunking and embeddings.\n")

    report = client.ingest(book)
    assert report.chunks_ingested > 0

    sources = client.sources()
    assert any(s.source_id == report.source_ids[0] for s in sources)

    results = client.retrieve("chunking", k=5)
    assert results
    assert all(r.chunk_id for r in results)


def test_ask_without_a_generator_raises(ragcore_client) -> None:
    client, _tmp_path = ragcore_client
    with pytest.raises(RagcoreClientError) as exc_info:
        client.ask("anything")
    assert exc_info.value.status_code == 400


def test_ask_with_a_generator_returns_a_result(ragcore_client_with_generator) -> None:
    client, tmp_path = ragcore_client_with_generator
    book = tmp_path / "book.md"
    book.write_text("# Chapter 1\n\nSome content about chunking.\n")
    client.ingest(book)

    result = client.ask("chunking")

    assert "answer to" in result.text
    assert result.citations


def test_delete_source_removes_its_chunks(ragcore_client) -> None:
    client, tmp_path = ragcore_client
    book = tmp_path / "book.md"
    book.write_text("# Chapter 1\n\nSome content.\n")
    report = client.ingest(book)

    deleted = client.delete_source(report.source_ids[0])

    assert deleted > 0
    assert client.sources() == []


def test_as_context_manager_closes_the_underlying_http_client(ragcore_client) -> None:
    client, _tmp_path = ragcore_client
    with client as c:
        assert c.health() is True
