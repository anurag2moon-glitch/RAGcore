"""VectorStore protocol: persisted vectors + metadata, queryable by
similarity and by the store-agnostic Filter DSL.

`get(chunk_ids)` is on the protocol even though vector-only retrieval never
calls it: the future graph layer needs to hydrate chunks by id when
expanding from graph neighbors back to text, and adding this method later
would be a breaking change for every store adapter written before then.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ragcore.types import Chunk, Filter, Retrieved, SourceInfo, Vector


@runtime_checkable
class VectorStore(Protocol):
    dim: int
    """Vector dimensionality this store was created for. Chunks embedded at
    a different dimension must be rejected by the implementation."""

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        """Insert or replace chunks (matched by chunk_id). Every chunk must
        have `vector` set."""
        ...

    def query(
        self, vector: Vector, k: int, filters: Filter | None = None
    ) -> list[Retrieved]:
        """Return the k nearest chunks to `vector`, optionally restricted by
        `filters`, ordered by descending score."""
        ...

    def get(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        """Hydrate chunks by id. Order need not match input order; callers
        that care re-key by chunk_id."""
        ...

    def delete(
        self,
        *,
        source_id: str | None = None,
        chunk_ids: Sequence[str] | None = None,
    ) -> int:
        """Delete by source_id (all chunks of that source) or by explicit
        chunk_ids. Exactly one of the two should be provided. Returns the
        number of chunks deleted."""
        ...

    def list_sources(self) -> list[SourceInfo]: ...

    def count(self) -> int: ...
