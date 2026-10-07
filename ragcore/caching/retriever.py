"""CachingRetriever — a Retriever decorator caching retrieve() results by
(query, k, filters), the same decorator seam every other Retriever layer
in this codebase uses (GraphRetriever, ConceptRetriever, ...):

    corpus.retriever = CachingRetriever(corpus.retriever, InMemoryResultCache())

A consumer who never wires this in sees no change at all.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from ragcore.caching.base import ResultCache
from ragcore.retrieval._filters import tenant_id_from_filter
from ragcore.types import Filter, Retrieved

if TYPE_CHECKING:
    from ragcore.retrieval.base import Retriever


def _cache_key(query: str, k: int, filters: Filter | None) -> str:
    """A stable hash over (query, k, filters) — sort_keys=True so two
    filters dicts with the same content in different insertion order
    (e.g. one built by TenantScopedCorpus's own dict-merge, which always
    puts tenant_id first) still hash identically."""
    payload = json.dumps({"query": query, "k": k, "filters": filters}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachingRetriever:
    def __init__(
        self,
        inner: Retriever,
        cache: ResultCache,
        *,
        ttl_seconds: float | None = None,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._ttl_seconds = ttl_seconds

    def retrieve(
        self, query: str, *, k: int = 10, filters: Filter | None = None
    ) -> list[Retrieved]:
        key = _cache_key(query, k, filters)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        results = self._inner.retrieve(query, k=k, filters=filters)
        self._cache.set(
            key,
            results,
            tenant_id=tenant_id_from_filter(filters),
            ttl_seconds=self._ttl_seconds,
        )
        return results
