"""LexicalStore protocol: a keyword/BM25 index alongside the VectorStore.

Shaped after ragcore.stores.base.VectorStore (index/search/delete/count) but
over text queries rather than vectors, and returning bare (chunk_id, rank)
hits rather than hydrated Chunks — ragcore.retrieval.hybrid.HybridRetriever
hydrates via Corpus.get()/VectorStore, the same way
ragcore.retrieval.graph.GraphRetriever already hydrates graph-expanded
chunk_ids. Keeping LexicalStore ignorant of Chunk hydration means a lexical
backend never needs to duplicate VectorStore's schema.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ragcore.lexical.types import LexicalHit
from ragcore.types import Chunk


@runtime_checkable
class LexicalStore(Protocol):
    def index(self, chunks: Sequence[Chunk]) -> None:
        """Insert or replace chunks (matched by chunk_id) in the keyword
        index. Only `chunk_id`, `text`, `meta.source_id`, and
        `meta.tenant_id` are used."""
        ...

    def search(
        self,
        query: str,
        k: int,
        *,
        source_ids: Sequence[str] | None = None,
        tenant_id: str | None = None,
    ) -> list[LexicalHit]:
        """Up to `k` keyword matches for `query`, best first. `source_ids`,
        when given, restricts the search to those sources — the fast path
        HybridRetriever uses for a source_id-only Filter; any other filter
        shape is applied by the caller after hydration, since a lexical
        backend generally can't evaluate the full Filter DSL. `tenant_id`,
        when given, is always safe to push down regardless of what else
        is in the caller's Filter — narrowing by it can never change which
        hits are correct, only how many irrelevant-tenant candidates get
        pulled in before the caller's own post-filter runs (see
        Fts5LexicalStore.search's docstring)."""
        ...

    def delete(
        self, *, source_id: str | None = None, chunk_ids: Sequence[str] | None = None
    ) -> int:
        """Delete by source_id (all chunks of that source) or by explicit
        chunk_ids. Exactly one of the two should be provided. Returns the
        number of chunks deleted."""
        ...

    def count(self) -> int: ...
