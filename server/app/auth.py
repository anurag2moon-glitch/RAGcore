"""API key auth: maps a request's API key to a tenant_id, via a
swappable ApiKeyStore. This is what Phase 7 item 30 (API keys/JWT/RBAC)
and docs/adr/0008-api-server-as-sibling-package.md both named as the
prerequisite for real multi-tenant HTTP routing — an unauthenticated
`X-Tenant-ID`-style header would have been the same false-security
shape as the Tier 2 shared-entity-id risk in docs/adr/0007, not actual
isolation.

**Opt-in, backward compatible**: if `RAGCORE_API_KEYS` is unset, auth
is disabled entirely and every route keeps operating exactly as before
this module existed — single-tenant, unauthenticated, over the raw
`Corpus`. `get_tenant_id()` returns `None` in that case; every call
site treats `None` as "no tenant scoping," never as "tenant none."
Setting `RAGCORE_API_KEYS` makes auth mandatory: every request without
a valid key is rejected, and every operation is scoped through
`ragcore.tenancy.TenantScopedCorpus`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol, runtime_checkable

from fastapi import Depends, HTTPException, Request

from app.env_parsing import parse_colon_pairs
from app.settings import get_settings


@runtime_checkable
class ApiKeyStore(Protocol):
    def resolve_tenant(self, api_key: str) -> str | None:
        """The tenant_id `api_key` is valid for, or None if the key is
        unknown/revoked."""
        ...


class InMemoryApiKeyStore:
    """Default ApiKeyStore: a plain dict, built once from
    RAGCORE_API_KEYS at process startup. A real deployment managing its
    own key lifecycle (issuance, rotation, revocation) implements this
    same Protocol against its own backing store — nothing else in this
    module or in main.py's routes needs to change for that swap."""

    def __init__(self, keys: dict[str, str]) -> None:
        self._keys = dict(keys)

    def resolve_tenant(self, api_key: str) -> str | None:
        return self._keys.get(api_key)


@lru_cache(maxsize=1)
def _api_key_store_singleton() -> ApiKeyStore | None:
    settings = get_settings()
    if not settings.ragcore_api_keys:
        return None
    return InMemoryApiKeyStore(parse_colon_pairs(settings.ragcore_api_keys, env_var="RAGCORE_API_KEYS"))


def get_api_key_store() -> ApiKeyStore | None:
    return _api_key_store_singleton()


def _extract_api_key(request: Request) -> str | None:
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        return auth_header[len("bearer "):].strip()
    return request.headers.get("x-api-key")


def get_tenant_id(
    request: Request,
    store: ApiKeyStore | None = Depends(get_api_key_store),  # noqa: B008 - FastAPI's own DI idiom
) -> str | None:
    if store is None:
        return None  # auth disabled - legacy single-tenant mode
    api_key = _extract_api_key(request)
    if api_key is None:
        raise HTTPException(
            status_code=401, detail="missing API key (Authorization: Bearer <key> or X-API-Key)"
        )
    tenant_id = store.resolve_tenant(api_key)
    if tenant_id is None:
        raise HTTPException(status_code=401, detail="invalid API key")
    return tenant_id
