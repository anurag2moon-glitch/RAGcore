"""Shared, store-agnostic evaluation of the Filter DSL (see ragcore.types.Filter)
directly against a ChunkMeta object in Python.

Every VectorStore backend translates the same DSL into its own query
language (see ragcore.stores.lancedb._translate_filters for the SQL form).
ragcore.retrieval.hybrid.HybridRetriever needs this Python-side form too: a
LexicalStore generally can't evaluate the DSL itself (SQLite FTS5 has no
general predicate language over UNINDEXED columns beyond simple equality),
so anything beyond a source_id-only filter is applied here, post-hydration.
This is the same logic tests/fakes.py's InMemoryStore has always used for
its own (dependency-free) VectorStore implementation, promoted here so it
has one canonical, tested home instead of being reinvented per caller.
"""

from __future__ import annotations

from typing import Any

from ragcore.types import ChunkMeta, Filter

_RANGE_OPS = {
    "gte": lambda a, v: a is not None and a >= v,
    "gt": lambda a, v: a is not None and a > v,
    "lte": lambda a, v: a is not None and a <= v,
    "lt": lambda a, v: a is not None and a < v,
    "eq": lambda a, v: a == v,
    "ne": lambda a, v: a != v,
}


def tenant_id_from_filter(filters: Filter | None) -> str | None:
    """Extracts a plain-string tenant_id value from `filters`, if present
    — alongside whatever else is in `filters`, unlike a narrower
    "this-key-only" extraction (e.g. ragcore.retrieval.hybrid's own
    source_id pushdown), since a tenant_id value is always safe to use
    as an *additional* narrowing constraint no matter what else the
    caller filtered on. Shared by every retrieval layer that reads its
    own auxiliary store (HybridRetriever's lexical channel,
    HierarchyRetriever's node candidates) and can push tenant_id down to
    that store directly, rather than relying solely on matches_filter()
    post-filtering the hydrated result (which still runs regardless, so
    this is a recall/performance narrowing, never the sole correctness
    mechanism — see docs/ARCHITECTURE.md Phase 7 gap item 28a)."""
    if not filters or "tenant_id" not in filters:
        return None
    value = filters["tenant_id"]
    return value if isinstance(value, str) else None


def matches_filter(meta: ChunkMeta, filters: Filter | None) -> bool:
    if not filters:
        return True
    for key, value in filters.items():
        actual: Any = getattr(meta, key)
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
