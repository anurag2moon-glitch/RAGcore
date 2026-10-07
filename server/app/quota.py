"""Per-tenant request quotas — the next piece of Phase 7 item 30, on
top of API keys (docs/adr/0012), rate limiting, and usage metering
already built.

**Deliberately a lifetime total cap per tenant, not a recurring
period-based quota** (daily/monthly) — a real period quota needs time-
windowing (and, for a multi-process deployment, a shared store), which
this reference in-memory implementation doesn't build; stated plainly
here rather than silently claimed. Reuses `app.usage.InMemoryUsageMeter`'s
counts directly rather than tracking its own separate total, so quota
enforcement and the `/usage` endpoint can never disagree about how much
a tenant has used.

Opt-in via `RAGCORE_TENANT_QUOTAS="tenant-a:1000,tenant-b:5000"` — a
tenant with no entry has no quota (unlimited); unset entirely means no
quota enforcement at all, the same backward-compatible shape every
other opt-in piece of this server uses.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, HTTPException

from app.auth import get_tenant_id
from app.env_parsing import parse_int_quotas
from app.settings import get_settings
from app.usage import InMemoryUsageMeter, get_usage_meter


@lru_cache(maxsize=1)
def _tenant_quotas_singleton() -> dict[str, int]:
    settings = get_settings()
    if not settings.ragcore_tenant_quotas:
        return {}
    return parse_int_quotas(settings.ragcore_tenant_quotas, env_var="RAGCORE_TENANT_QUOTAS")


def get_tenant_quotas() -> dict[str, int]:
    return _tenant_quotas_singleton()


def enforce_quota(
    meter: InMemoryUsageMeter = Depends(get_usage_meter),  # noqa: B008 - FastAPI's own DI idiom
    tenant_id: str | None = Depends(get_tenant_id),
    quotas: dict[str, int] = Depends(get_tenant_quotas),  # noqa: B008
) -> None:
    """Must run *before* `app.usage.record_usage` in a route's
    dependency list — it checks usage *before* this request, so the
    request that would exceed the quota is rejected rather than
    counted. A tenant at exactly `limit - 1` used requests is still
    allowed (the request that brings them to `limit`); the next one is
    rejected. No-op in single-tenant mode or when no quota is
    configured for this tenant."""
    if tenant_id is None:
        return
    limit = quotas.get(tenant_id)
    if limit is None:
        return
    used = sum(meter.usage_for(tenant_id).values())
    if used >= limit:
        raise HTTPException(
            status_code=402, detail=f"quota exceeded: {used}/{limit} lifetime requests used"
        )
