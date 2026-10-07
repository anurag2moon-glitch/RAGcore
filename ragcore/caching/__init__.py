"""Phase 7 gap item 31: a semantic/result cache with tenant-scoped
invalidation. See ragcore.caching.base.ResultCache, ragcore.caching.memory.
InMemoryResultCache (the default, zero-new-dependency implementation —
Redis is deliberately left for the Phase 8 server layer, see that
module's docstring), and ragcore.caching.retriever.CachingRetriever (the
Retriever decorator that actually uses one).
"""

from ragcore.caching.base import ResultCache
from ragcore.caching.memory import InMemoryResultCache
from ragcore.caching.retriever import CachingRetriever

__all__ = ["CachingRetriever", "InMemoryResultCache", "ResultCache"]
