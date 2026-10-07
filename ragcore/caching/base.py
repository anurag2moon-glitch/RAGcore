"""ResultCache protocol: caches a Retriever's output keyed by (query, k,
filters), with tenant-scoped invalidation as a first-class operation —
not just a TTL side effect.

Deliberately read-only from Corpus/TenantScopedCorpus's point of view:
nothing in ragcore auto-invalidates a cache when a corpus is mutated
(ingest, remove, hard_delete, build_graph, ...). Corpus has too many
mutation paths for ragcore to safely guess which ones should trigger
which cache's invalidation — a consumer who wires a ResultCache onto a
Retriever is responsible for calling invalidate() after whatever
mutations it cares about, the same opt-in, nothing-automatic discipline
every other decorator in this codebase follows.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragcore.types import Retrieved


@runtime_checkable
class ResultCache(Protocol):
    def get(self, key: str) -> list[Retrieved] | None:
        """None on a miss (including an expired entry) — never raises."""
        ...

    def set(
        self,
        key: str,
        results: list[Retrieved],
        *,
        tenant_id: str | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        """`tenant_id`, when given, tags this entry so a later
        `invalidate(tenant_id=...)` call can target it specifically.
        `ttl_seconds=None` means no expiry (cleared only by an explicit
        `invalidate()` call or `clear()`)."""
        ...

    def invalidate(self, *, tenant_id: str | None = None) -> int:
        """Removes every entry tagged with `tenant_id` (or, if
        `tenant_id=None`, every entry regardless of tag) — the
        tenant-scoped invalidation mechanism. Returns the number of
        entries removed."""
        ...
