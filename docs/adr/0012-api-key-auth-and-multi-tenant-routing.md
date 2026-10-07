# 12. API key auth gates real multi-tenant HTTP routing

Date: 2026-10-07

## Status

Accepted

## Context

`docs/adr/0008-api-server-as-sibling-package.md` shipped `server/` as
deliberately single-tenant, naming the exact reason: multi-tenant HTTP
routing needs a real way to know *which* tenant is calling, and an
unauthenticated header a caller could set to any value (e.g.
`X-Tenant-ID`) would be the same false-security shape as the Tier 2
shared-entity-id risk `docs/adr/0007-tenant-id-on-remaining-stores-two-tiers.md`
describes — isolation that looks like it exists but doesn't. Phase 7
gap item 30 (API keys/JWT/RBAC) and item 39 (multi-tenant HTTP routing)
were both left open, item 39 explicitly gated on item 30.

`ragcore.tenancy.TenantScopedCorpus` already exists and already does
the enforcement work (commit 55f5410) — what was missing was a
trustworthy way for an HTTP request to say which tenant it's acting
as.

## Decision

`server/app/auth.py` adds an `ApiKeyStore` Protocol
(`resolve_tenant(api_key) -> str | None`) and a default
`InMemoryApiKeyStore`, loaded once at startup from a `RAGCORE_API_KEYS`
environment variable (`"key1:tenant-a,key2:tenant-b"` — a reference
implementation's worth of key management, not a production one; a real
deployment implements the same Protocol against its own key-issuance
backend). A `get_tenant_id` FastAPI dependency extracts the key from
`Authorization: Bearer <key>` or `X-API-Key`, resolves it, and raises
`401` if missing or invalid.

**Opt-in, backward compatible, the same discipline every other
addition in this codebase follows**: if `RAGCORE_API_KEYS` is unset,
`get_tenant_id` returns `None`, and `server/app/corpus_provider.py`'s
new `get_scoped_corpus` dependency hands every route the raw,
unscoped `Corpus` — byte-for-byte the server's behavior before this
ADR. Setting `RAGCORE_API_KEYS` makes auth mandatory and every route
receives a fresh `TenantScopedCorpus(corpus, tenant_id)` instead,
which forces that tenant_id on every call it makes — the enforcement
logic lives entirely in `ragcore.tenancy`, not duplicated here.

A `TenancyError` (raised by `TenantScopedCorpus` when an operation
would touch another tenant's data — e.g. `DELETE
/sources/{source_id}` for a source that isn't yours) gets its own
exception handler returning `403`, registered ahead of the general
`RagcoreError` → `400` handler, since "you don't own this" is a
meaningfully different condition from a generic bad request.

## Consequences

- Closes Phase 7 items 30 and 39 for the reference server specifically
  — a real, working, tested multi-tenant HTTP surface exists now, not
  just the mechanism `TenantScopedCorpus` already provided.
- **A real pre-existing gap in `server/tests/fakes.py`'s
  `InMemoryStore.query()` was caught while testing this**: it ignored
  the `filters` argument entirely, unlike the equivalent fake in the
  main `tests/fakes.py`. Every server test written before this ADR
  happened to never rely on filtering, so this went unnoticed; the new
  cross-tenant isolation test
  (`test_retrieve_never_returns_another_tenants_chunks`,
  `server/tests/test_auth.py`) failed immediately against the
  unfixed fake — a real tool catching a real gap, not a hypothetical
  one. Fixed by reusing `ragcore.retrieval._filters.matches_filter`,
  the same Python-side Filter DSL evaluator every other in-memory
  post-hydration filtering path in this codebase already uses, rather
  than writing a third reimplementation.
- `server/`'s `InMemoryApiKeyStore` is explicitly a reference
  implementation, not production key management — no key rotation, no
  expiry, no hashing of keys at rest (keys are compared in plaintext
  from the env var). A real deployment should implement `ApiKeyStore`
  against its own backend before using this in production, same as
  `InMemoryResultCache`/`InMemoryTranscriptStore` are explicitly
  positioned as per-process defaults elsewhere in this codebase
  family, not as a claim that this is production-grade auth.
- `TenantScopedCorpus`'s own documented scope limit still applies
  unchanged: `GraphRetriever`/`ConceptRetriever` read Tier 2 stores
  with no `tenant_id` at all (item 28b). A multi-tenant deployment of
  `server/` must not wire either of those retrieval layers onto the
  shared `Corpus` until that design question is resolved — this ADR
  does not change that constraint, only adds enforcement for what
  `TenantScopedCorpus` already covers.

**Follow-up, same day**: `server/app/rate_limit.py` adds the
remaining straightforward piece of item 30 — a per-key (tenant_id when
auth is on, else client IP) token-bucket rate limiter, opt-in via
`RAGCORE_RATE_LIMIT_PER_MINUTE`, applied as a per-route dependency so
`/health` stays exempt. This didn't warrant its own ADR — a standard
token bucket keyed by the tenant_id this same ADR's auth layer already
resolves, no new design fork.

**Follow-up, same day**: `server/app/usage.py` adds request-count
usage metering (`GET /usage`, a tenant's own counts only), the
"billing-ready usage metering" phrase from item 30 — deliberately
scoped to request counts, not token/cost. This server shares one
embedder across every tenant, so attributing its token usage back to
individual tenants isn't well-defined without redesigning that
sharing — a real limitation stated plainly in `app/usage.py`'s own
docstring rather than left for a reader to discover. Request-count
metering is still genuinely useful (pay-per-request tiers, abuse
detection) and composes with the real token-cost tracking that already
exists (`ragcore.embedding.cost.UsageTracker`) if a future redesign
gives each tenant its own embedder instance.

**Follow-up, same day**: `server/app/quota.py` adds per-tenant quotas
(`RAGCORE_TENANT_QUOTAS`), the last tractable piece of item 30 this
pass tackles. Deliberately a **lifetime total** cap, not a recurring
period-based (daily/monthly) quota — a real period quota needs time-
windowing this reference in-memory implementation doesn't build,
stated plainly in `app/quota.py`'s own docstring rather than silently
claimed. Reuses `app.usage.InMemoryUsageMeter`'s counts directly
rather than tracking a separate total, so quota enforcement and
`GET /usage` can never disagree about how much a tenant has used. The
quota check runs before `record_usage` in each route's dependency
list specifically so a rejected request is never counted — verified by
a dedicated test, not assumed from the ordering alone. JWT/OAuth and
RBAC remain the open parts of item 30.
