"""Per-tenant token-cost quota for rule extraction jobs — Phase 10 §8's
"quotas apply to extraction jobs (token cost metering)", distinct from
`app.quota`'s existing per-request quota (which meters every route
uniformly by *request count*, not cost). Extraction is the one
operation in this server whose cost varies enormously request-to-
request (a one-chapter corpus vs. a 40-chapter one), which is exactly
why it gets its own token-based cap rather than reusing the generic
request-count one.

**Checked before, recorded after**: unlike a request-count quota
(whose cost — one request — is known up front), an extraction job's
token cost isn't known until `build_rules()` actually returns
(`RuleExtractionReport.total_tokens_used`). `enforce_extraction_token_quota`
rejects a request if the tenant is *already* over budget from past
jobs; it cannot pre-reject based on this job's own cost, which isn't
known yet. A single job can therefore push a tenant over their quota
by however much that one job costs — the same "the request that
brings you to the limit is still allowed, only the next one is
rejected" shape `app.quota.enforce_quota` already has, just unable to
be exact about the overshoot amount here.

Opt-in via `RAGCORE_EXTRACTION_TOKEN_QUOTAS="tenant-a:100000,tenant-b:500000"`
— a tenant with no entry is unlimited; unset entirely means no
enforcement at all, same backward-compatible shape every other opt-in
piece of this server uses.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache

from fastapi import Depends, HTTPException

from app.auth import get_tenant_id

RAGCORE_EXTRACTION_TOKEN_QUOTAS_ENV = "RAGCORE_EXTRACTION_TOKEN_QUOTAS"


def _parse_quotas_env(value: str) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for pair in value.split(","):
        pair = pair.strip()
        if not pair:
            continue
        tenant_id, _, limit_str = pair.partition(":")
        if not tenant_id or not limit_str:
            raise RuntimeError(
                f"{RAGCORE_EXTRACTION_TOKEN_QUOTAS_ENV} entry {pair!r} is not in "
                f"'tenant_id:limit' form"
            )
        try:
            limit = int(limit_str)
        except ValueError:
            raise RuntimeError(
                f"{RAGCORE_EXTRACTION_TOKEN_QUOTAS_ENV} entry {pair!r} has a non-integer limit"
            ) from None
        quotas[tenant_id] = limit
    return quotas


@lru_cache(maxsize=1)
def _extraction_token_quotas_singleton() -> dict[str, int]:
    raw = os.environ.get(RAGCORE_EXTRACTION_TOKEN_QUOTAS_ENV)
    if not raw:
        return {}
    return _parse_quotas_env(raw)


def get_extraction_token_quotas() -> dict[str, int]:
    return _extraction_token_quotas_singleton()


class InMemoryExtractionTokenMeter:
    """Lifetime total tokens consumed by rule extraction jobs, per
    tenant. In-memory only, like every other reference meter in this
    server — a real deployment metering real spend needs a persistent,
    shared backend across processes; stated plainly, not silently
    assumed durable."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens_by_tenant: dict[str, int] = {}

    def tokens_used(self, tenant_id: str) -> int:
        with self._lock:
            return self._tokens_by_tenant.get(tenant_id, 0)

    def record(self, tenant_id: str, tokens: int) -> None:
        with self._lock:
            self._tokens_by_tenant[tenant_id] = self._tokens_by_tenant.get(tenant_id, 0) + tokens


@lru_cache(maxsize=1)
def _extraction_token_meter_singleton() -> InMemoryExtractionTokenMeter:
    return InMemoryExtractionTokenMeter()


def get_extraction_token_meter() -> InMemoryExtractionTokenMeter:
    return _extraction_token_meter_singleton()


def enforce_extraction_token_quota(
    meter: InMemoryExtractionTokenMeter = Depends(get_extraction_token_meter),  # noqa: B008
    tenant_id: str | None = Depends(get_tenant_id),
    quotas: dict[str, int] = Depends(get_extraction_token_quotas),  # noqa: B008
) -> None:
    if tenant_id is None:
        return
    limit = quotas.get(tenant_id)
    if limit is None:
        return
    used = meter.tokens_used(tenant_id)
    if used >= limit:
        raise HTTPException(
            status_code=402,
            detail=f"extraction token quota exceeded: {used}/{limit} lifetime tokens used",
        )


def record_extraction_tokens(
    tokens: int, *, meter: InMemoryExtractionTokenMeter, tenant_id: str | None
) -> None:
    """A plain function, not a FastAPI dependency (it takes `tokens`,
    which no request ever supplies) — called directly by the route
    handler after `build_rules()` returns, once the real cost is
    known, passing the `meter`/`tenant_id` it already resolved via its
    own `Depends(...)` parameters. Single-tenant mode (`tenant_id is
    None`) records nothing: there is no per-tenant quota to meter in
    that mode."""
    if tenant_id is None:
        return
    meter.record(tenant_id, tokens)
