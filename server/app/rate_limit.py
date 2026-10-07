"""Per-key rate limiting — the remaining piece of Phase 7 item 30 this
pass tackles (API keys landed in docs/adr/0012; JWT/OAuth/RBAC/
billing-metering stay open).

Opt-in via `RAGCORE_RATE_LIMIT_PER_MINUTE` — unset means no rate
limiting at all, identical to this server's behavior before this
module existed, the same backward-compatible shape `app.auth` uses for
`RAGCORE_API_KEYS`.

Keys by the resolved tenant_id when multi-tenant auth
(`RAGCORE_API_KEYS`) is enabled; otherwise by client IP — a rate
limiter is still useful in single-tenant mode (protecting one `Corpus`
from one noisy client), it just has no tenant_id to key by in that
case.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

from fastapi import Depends, HTTPException, Request

from app.auth import get_tenant_id

RAGCORE_RATE_LIMIT_PER_MINUTE_ENV = "RAGCORE_RATE_LIMIT_PER_MINUTE"


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class InMemoryRateLimiter:
    """Token bucket per key: starts full (a caller can burst up to
    `requests_per_minute` immediately), refills continuously at
    `requests_per_minute / 60` tokens/sec, capped at
    `requests_per_minute` — standard token-bucket semantics, not a
    fixed-window counter (which would let a caller burst 2x the
    configured rate across a window boundary)."""

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._rate_per_second = requests_per_minute / 60.0
        self._capacity = float(requests_per_minute)
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=self._capacity, last_refill=now)
                self._buckets[key] = bucket
            elapsed = now - bucket.last_refill
            bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._rate_per_second)
            bucket.last_refill = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True
            return False


@lru_cache(maxsize=1)
def _rate_limiter_singleton() -> InMemoryRateLimiter | None:
    raw = os.environ.get(RAGCORE_RATE_LIMIT_PER_MINUTE_ENV)
    if not raw:
        return None
    return InMemoryRateLimiter(int(raw))


def get_rate_limiter() -> InMemoryRateLimiter | None:
    return _rate_limiter_singleton()


def enforce_rate_limit(
    request: Request,
    limiter: InMemoryRateLimiter | None = Depends(get_rate_limiter),  # noqa: B008 - FastAPI's own DI idiom
    tenant_id: str | None = Depends(get_tenant_id),
) -> None:
    """A route dependency, not a global ASGI middleware — applied
    per-route in main.py so `/health` can stay exempt. `tenant_id` is
    re-requested here even though `get_scoped_corpus` also depends on
    `get_tenant_id`; FastAPI caches a dependency's result per request
    by default, so this doesn't re-validate the API key twice."""
    if limiter is None:
        return
    key = tenant_id if tenant_id is not None else _client_key(request)
    if not limiter.allow(key):
        raise HTTPException(status_code=429, detail="rate limit exceeded")


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"
