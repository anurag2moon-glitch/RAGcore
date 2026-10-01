"""EmbeddingCache — a content-hash + model-version cache for Corpus's
core embed stage (docs/ARCHITECTURE.md Phase 2 gap list item 10:
"embedding cache keyed by content hash plus model version... versioned
embeddings, zero-downtime re-embedding migrations").

Keyed on (content_hash, model_name, dim), not content_hash alone: two
Embedders can share a model name but a different output dimension
(OpenAIEmbedder's `dim=`, VoyageContextEmbedder's `output_dimension=`),
which produces genuinely different vectors — caching on name alone risks
handing back a wrong-dimension vector to whichever VectorStore the caller
is populating.

Zero-downtime re-embedding migration falls out of the key shape, rather
than needing a separate mechanism: embedding the same corpus under a new
model (different name or dim) is a guaranteed cache miss for every chunk,
so the old model's cached vectors are untouched and still servable (e.g.
by a Corpus still pointed at the old model, serving live traffic) while
the new model's embed run populates its own, disjoint cache entries
alongside them.

Not part of the Embedder protocol — this caches at the granularity
Corpus._embed_groups() controls (whole ChunkGroups, to preserve
contextualization integrity; see Corpus._embed_groups_with_cache's
docstring), not inside a provider implementation.
"""

from __future__ import annotations

import sqlite3
import struct
import threading
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from ragcore.types import Vector

CacheKey = tuple[str, str, int]
"""(content_hash, model_name, dim)."""


@runtime_checkable
class EmbeddingCache(Protocol):
    def get_many(self, keys: Sequence[CacheKey]) -> dict[CacheKey, Vector]:
        """Whichever of `keys` are present, as a dict — callers check
        `key in result` for a miss, never assume every key comes back."""
        ...

    def put_many(self, items: dict[CacheKey, Vector]) -> None:
        """Insert or replace. A re-write of an existing key with the same
        vector (e.g. two ingests of the same unchanged content) is a
        harmless no-op in effect, not an error."""
        ...


def _pack(vector: Vector) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes) -> Vector:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash TEXT NOT NULL,
    model_name TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (content_hash, model_name, dim)
);
"""


class SqliteEmbeddingCache:
    """Default EmbeddingCache: one SQLite file, vectors packed as
    little-endian float32 blobs (half the size of float64, and plenty of
    precision for a cached embedding that's about to feed a cosine/dot
    similarity search, not a numerically sensitive computation)."""

    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)

    def get_many(self, keys: Sequence[CacheKey]) -> dict[CacheKey, Vector]:
        if not keys:
            return {}
        out: dict[CacheKey, Vector] = {}
        # SQLite's default compiled limit on bound parameters (~32766 in
        # modern builds, historically as low as 999) makes one unbounded
        # IN-list risky at scale; chunking keeps this safe at any corpus size.
        for batch in _chunked(list(keys), 500):
            placeholders = ",".join(["(?,?,?)"] * len(batch))
            params = [v for key in batch for v in key]
            rows = self._conn.execute(
                "SELECT content_hash, model_name, dim, vector FROM embedding_cache "
                f"WHERE (content_hash, model_name, dim) IN ({placeholders})",
                params,
            ).fetchall()
            for content_hash, model_name, dim, blob in rows:
                out[(content_hash, model_name, dim)] = _unpack(blob)
        return out

    def put_many(self, items: dict[CacheKey, Vector]) -> None:
        if not items:
            return
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.executemany(
                    "INSERT INTO embedding_cache (content_hash, model_name, dim, vector) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(content_hash, model_name, dim) DO UPDATE SET "
                    "vector = excluded.vector",
                    [
                        (content_hash, model_name, dim, _pack(vector))
                        for (content_hash, model_name, dim), vector in items.items()
                    ],
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()
        return row[0]


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]
