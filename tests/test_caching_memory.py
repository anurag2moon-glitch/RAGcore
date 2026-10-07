"""InMemoryResultCache: get/set/invalidate, TTL expiry, tenant-scoped
invalidation, and the hit/miss counters.
"""

from __future__ import annotations

import time

from ragcore.caching.memory import InMemoryResultCache
from ragcore.types import Chunk, ChunkMeta, Retrieved


def _result(chunk_id: str) -> list[Retrieved]:
    meta = ChunkMeta(
        source_id="s", source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=None, section=None,
        page_start=None, page_end=None, chunk_index=0, context_group_id="g",
        char_start=0, char_end=4, token_count=1,
    )
    return [Retrieved(chunk=Chunk(chunk_id=chunk_id, text="text", meta=meta), score=0.9)]


def test_miss_on_an_unknown_key_returns_none() -> None:
    cache = InMemoryResultCache()
    assert cache.get("nonexistent") is None
    assert cache.misses == 1
    assert cache.hits == 0


def test_set_then_get_returns_the_same_results() -> None:
    cache = InMemoryResultCache()
    results = _result("c1")
    cache.set("k1", results)

    got = cache.get("k1")

    assert got is not None
    assert [r.chunk.chunk_id for r in got] == ["c1"]
    assert cache.hits == 1


def test_ttl_expiry_makes_a_later_get_miss() -> None:
    cache = InMemoryResultCache()
    cache.set("k1", _result("c1"), ttl_seconds=0.01)

    assert cache.get("k1") is not None
    time.sleep(0.02)
    assert cache.get("k1") is None


def test_no_ttl_never_expires() -> None:
    cache = InMemoryResultCache()
    cache.set("k1", _result("c1"), ttl_seconds=None)
    time.sleep(0.02)
    assert cache.get("k1") is not None


def test_invalidate_with_no_tenant_id_clears_everything() -> None:
    cache = InMemoryResultCache()
    cache.set("k1", _result("c1"), tenant_id="tenant-a")
    cache.set("k2", _result("c2"), tenant_id="tenant-b")

    removed = cache.invalidate()

    assert removed == 2
    assert cache.get("k1") is None
    assert cache.get("k2") is None


def test_invalidate_with_a_tenant_id_only_removes_that_tenants_entries() -> None:
    cache = InMemoryResultCache()
    cache.set("k1", _result("c1"), tenant_id="tenant-a")
    cache.set("k2", _result("c2"), tenant_id="tenant-b")
    cache.set("k3", _result("c3"), tenant_id=None)

    removed = cache.invalidate(tenant_id="tenant-a")

    assert removed == 1
    assert cache.get("k1") is None
    assert cache.get("k2") is not None
    assert cache.get("k3") is not None


def test_len_reflects_current_entry_count() -> None:
    cache = InMemoryResultCache()
    cache.set("k1", _result("c1"))
    cache.set("k2", _result("c2"))
    assert len(cache) == 2
    cache.invalidate(tenant_id=None)
    assert len(cache) == 0
