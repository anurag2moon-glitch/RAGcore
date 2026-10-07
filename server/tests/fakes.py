"""Shared in-memory test doubles for server/'s tests — no network, no
real Voyage/LanceDB credentials. Used by both server/tests/test_api.py
(FastAPI's TestClient, in-process) and run_test_server.py (a real
uvicorn process other sibling packages' test suites, e.g.
sdk/typescript/'s, spawn to exercise the HTTP layer over a real socket)
— imported directly in both places rather than redefined, so they can't
silently drift apart.
"""

from __future__ import annotations

from ragcore.generation.base import Answer
from ragcore.retrieval._filters import matches_filter
from ragcore.types import Retrieved, SourceInfo


class FakeEmbedder:
    dim = 8
    model = "fake-embedder"

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.dim

    def embed_groups(self, groups, *, input_type="document"):
        return [[[0.1] * self.dim for _ in g] for g in groups]


class InMemoryStore:
    dim = 8

    def __init__(self) -> None:
        self._chunks: dict[str, object] = {}

    def upsert(self, chunks) -> None:
        for c in chunks:
            self._chunks[c.chunk_id] = c

    def query(self, vector, k, filters=None):
        """Filters, unlike an earlier version of this fake which ignored
        `filters` entirely — a real gap this fake previously had,
        caught by test_retrieve_never_returns_another_tenants_chunks
        (server/tests/test_auth.py) actually failing against it, not
        assumed to already be correct. Reuses
        ragcore.retrieval._filters.matches_filter, the same Python-side
        Filter DSL evaluator every other in-memory/post-hydration
        filtering path in this codebase uses, rather than a third
        reimplementation alongside tests/fakes.py's own `_matches`."""
        candidates = list(self._chunks.values())
        if filters:
            candidates = [c for c in candidates if matches_filter(c.meta, filters)]
        return [Retrieved(chunk=c, score=0.9) for c in candidates[:k]]

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


class FakeGenerator:
    model = "fake-generator"

    def generate(self, query, context) -> Answer:
        return Answer(text=f"answer to {query!r}", citations=list(context))
