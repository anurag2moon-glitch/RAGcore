"""Per-tenant request-count usage metering — a further piece of Phase 7
item 30 ("billing-ready usage metering"), on top of the auth
(docs/adr/0012) and rate-limiting layers already built.

Scoped deliberately to request counts per endpoint, not token/cost
metering: token-level cost (ragcore.embedding.cost.UsageTracker already
exists for this) would need the embedder itself to be tenant-aware,
but this server shares one embedder across every tenant — attributing
one shared embedder's token usage back to individual tenants isn't
well-defined without a bigger redesign, so this stays a request-count
meter, the metering unit "billing-ready" most directly and honestly
supports today. A GET /usage endpoint lets a tenant read only its own
counts — never another tenant's.

Opt-in only in the sense that it always runs once auth is enabled
(RAGCORE_API_KEYS) — metering without knowing *whose* request it is
would have nothing useful to record, so this has no separate on/off
switch of its own; in single-tenant mode (no auth), there is no
tenant_id to meter by and /usage is unavailable.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from functools import lru_cache

from fastapi import Depends, HTTPException, Request

from app.auth import get_tenant_id


class InMemoryUsageMeter:
    """Reference implementation: per-process, per-tenant request
    counters, keyed by (tenant_id, endpoint). A real deployment
    persists this (Postgres, Redis, ...) rather than losing counts on
    every restart — same "reference, not production" positioning as
    `InMemoryApiKeyStore`/`InMemoryRateLimiter`."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], int] = defaultdict(int)
        self._lock = threading.Lock()

    def record(self, tenant_id: str, endpoint: str) -> None:
        with self._lock:
            self._counts[(tenant_id, endpoint)] += 1

    def usage_for(self, tenant_id: str) -> dict[str, int]:
        with self._lock:
            return {
                endpoint: count
                for (tid, endpoint), count in self._counts.items()
                if tid == tenant_id
            }


@lru_cache(maxsize=1)
def _usage_meter_singleton() -> InMemoryUsageMeter:
    return InMemoryUsageMeter()


def get_usage_meter() -> InMemoryUsageMeter:
    return _usage_meter_singleton()


def record_usage(
    request: Request,
    meter: InMemoryUsageMeter = Depends(get_usage_meter),  # noqa: B008 - FastAPI's own DI idiom
    tenant_id: str | None = Depends(get_tenant_id),
) -> None:
    """A route dependency, applied the same way `enforce_rate_limit`
    is — per-route, so `/health` and `/usage` itself stay unmetered.
    A no-op in single-tenant mode (no tenant_id to attribute the
    request to)."""
    if tenant_id is None:
        return
    meter.record(tenant_id, f"{request.method} {request.url.path}")


def get_my_usage(
    meter: InMemoryUsageMeter = Depends(get_usage_meter),  # noqa: B008
    tenant_id: str | None = Depends(get_tenant_id),
) -> dict[str, int]:
    """Returns only the calling tenant's own usage — never another
    tenant's, since the only identity this has to key by is the one
    `get_tenant_id` already authenticated this exact request as."""
    if tenant_id is None:
        raise HTTPException(
            status_code=400,
            detail="usage metering requires multi-tenant mode (RAGCORE_API_KEYS)",
        )
    return meter.usage_for(tenant_id)
