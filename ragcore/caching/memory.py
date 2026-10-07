"""InMemoryResultCache — the default ResultCache: a plain dict behind a
threading.Lock, zero new dependency (no Redis — see ragcore/caching's
package docstring for why that's deliberately left for the Phase 8 server
layer, where a cache actually needs to be shared across processes).

Per-process, per-instance only: exactly the same "ephemeral, session-
scoped" characterization ragcore.conversation.session.InMemoryTranscriptStore
gives its own default — a horizontally-scaled deployment needs a shared
backend, implementing the same ResultCache Protocol.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from ragcore.types import Retrieved


@dataclass
class _Entry:
    results: list[Retrieved]
    tenant_id: str | None
    expires_at: float | None


class InMemoryResultCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> list[Retrieved] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.expires_at is not None and time.monotonic() >= entry.expires_at:
                del self._entries[key]
                self.misses += 1
                return None
            self.hits += 1
            return list(entry.results)

    def set(
        self,
        key: str,
        results: list[Retrieved],
        *,
        tenant_id: str | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        expires_at = time.monotonic() + ttl_seconds if ttl_seconds is not None else None
        with self._lock:
            self._entries[key] = _Entry(
                results=list(results), tenant_id=tenant_id, expires_at=expires_at
            )

    def invalidate(self, *, tenant_id: str | None = None) -> int:
        with self._lock:
            if tenant_id is None:
                count = len(self._entries)
                self._entries.clear()
                return count
            to_remove = [k for k, e in self._entries.items() if e.tenant_id == tenant_id]
            for k in to_remove:
                del self._entries[k]
            return len(to_remove)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
