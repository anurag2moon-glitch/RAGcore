"""Test doubles for Embedder, VectorStore, and Generator.

These exist to prove the protocols are honest: the whole ingest -> retrieve
-> ask pipeline should run against these with no network access and no
native dependencies (no lancedb, no voyageai, no anthropic). FakeEmbedder
is deterministic (hash-based) so the same text always produces the same
vector — useful for chunk-id-stability and store-swap comparisons, not for
semantic relevance. FakeGenerator is likewise deterministic and just
echoes back what it was given, so tests can assert on composition (did
ask() pass retrieve()'s results through?) without asserting on prose.
"""

from __future__ import annotations

import hashlib
import math
from typing import Literal, Sequence

from ragcore.generation.base import Answer
from ragcore.types import Chunk, Filter, Retrieved, SourceInfo, Vector


class FakeEmbedder:
    name = "fake-hash-embedder"

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def _vec(self, text: str) -> Vector:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [(h[i % len(h)] / 127.5) - 1.0 for i in range(self.dim)]

    def embed_groups(
        self,
        groups: Sequence[Sequence[str]],
        *,
        input_type: Literal["document", "query", None] = "document",
    ) -> list[list[Vector]]:
        return [[self._vec(t) for t in group] for group in groups]

    def embed_query(self, text: str) -> Vector:
        return self._vec(text)


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


_RANGE_OPS = {
    "gte": lambda a, v: a is not None and a >= v,
    "gt": lambda a, v: a is not None and a > v,
    "lte": lambda a, v: a is not None and a <= v,
    "lt": lambda a, v: a is not None and a < v,
    "eq": lambda a, v: a == v,
    "ne": lambda a, v: a != v,
}


def _matches(meta: object, filters: Filter) -> bool:
    for key, value in filters.items():
        actual = getattr(meta, key)
        if isinstance(value, dict):
            if not all(_RANGE_OPS[op](actual, operand) for op, operand in value.items()):
                return False
        elif isinstance(value, (list, tuple, set)):
            if actual not in value:
                return False
        else:
            if actual != value:
                return False
    return True


class InMemoryStore:
    """VectorStore implementation with no persistence and no native deps.
    Uses cosine similarity to match LanceDBStore's default metric, so the
    two are comparable in the store-swap test.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._chunks: dict[str, Chunk] = {}

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        for c in chunks:
            self._chunks[c.chunk_id] = c

    def query(self, vector: Vector, k: int, filters: Filter | None = None) -> list[Retrieved]:
        candidates = list(self._chunks.values())
        if filters:
            candidates = [c for c in candidates if _matches(c.meta, filters)]
        scored = [(c, _cosine(vector, c.vector)) for c in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [Retrieved(chunk=c, score=s) for c, s in scored[:k]]

    def get(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return [self._chunks[i] for i in chunk_ids if i in self._chunks]

    def delete(self, *, source_id: str | None = None, chunk_ids: Sequence[str] | None = None) -> int:
        before = len(self._chunks)
        if source_id is not None:
            self._chunks = {
                cid: c for cid, c in self._chunks.items() if c.meta.source_id != source_id
            }
        else:
            for cid in chunk_ids or []:
                self._chunks.pop(cid, None)
        return before - len(self._chunks)

    def list_sources(self) -> list[SourceInfo]:
        counts: dict[str, int] = {}
        first: dict[str, Chunk] = {}
        for c in self._chunks.values():
            counts[c.meta.source_id] = counts.get(c.meta.source_id, 0) + 1
            first.setdefault(c.meta.source_id, c)
        return [
            SourceInfo(
                source_id=sid,
                path=first[sid].meta.source_path,
                title=first[sid].meta.source_title,
                source_type=first[sid].meta.source_type,
                chunk_count=n,
            )
            for sid, n in counts.items()
        ]

    def count(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> list[Chunk]:
        """Test-only convenience, not part of the VectorStore protocol."""
        return list(self._chunks.values())


class FakeGenerator:
    """Deterministic Generator double: no network, no LLM. Echoes the query
    and the ordered chunk_ids it was given, so a test can assert on
    composition (what retrieve() produced flowing correctly into generate())
    without asserting on real prose."""

    model = "fake-echo-generator"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def generate(self, query: str, context: Sequence[Retrieved]) -> Answer:
        chunk_ids = [r.chunk.chunk_id for r in context]
        self.calls.append((query, chunk_ids))
        text = f"[fake answer to {query!r} using {len(chunk_ids)} chunk(s): {chunk_ids}]"
        return Answer(text=text, citations=list(context))
