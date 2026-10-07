# 8. The reference API server is a sibling package, not part of ragcore

Date: 2026-10-07

## Status

Accepted

## Context

Phase 8 of the original spec calls for "FastAPI service, OpenAPI spec,
gRPC, Python/TypeScript SDKs, MCP server, Docker Compose/Helm/K8s
manifests, Redis, feature flags, load tests, backup/restore runbooks" —
and `docs/ARCHITECTURE.md`'s own gap list says Phase 8 is "everything,
from scratch, after Phase 7's tenancy model is settled." Phase 7 is now
done for everything tractable as library code (see Phase 7's section);
the remaining Phase 7 items (API keys/JWT/RBAC/quotas/rate-limiting,
encryption at rest/audit logs) explicitly need an actual request/server
context, which is exactly what Phase 8 builds.

`CLAUDE.md`'s own stated design boundary for `ragcore` is explicit:
"Library, not application — no CLI, no server, no global state."
Building a FastAPI service *inside* `ragcore/` would directly violate
that boundary, and would force every consumer of the library (including
Mimisi and Vastu Shastra, the two downstream products this base serves)
to pull in `fastapi`/`uvicorn` as dependencies whether or not they ever
run a server.

## Decision

The reference API server lives in a new top-level `server/` directory,
sibling to `ragcore/` in this same repository — not inside the `ragcore`
package, and not a separate repository (keeping it in-repo, for now,
means one `git clone` gets both the library and its reference server,
and CI can exercise both together). It has its own `server/pyproject.toml`
declaring `fastapi`/`uvicorn` as *its* dependencies; `ragcore` itself is
installed into the same venv separately (`pip install -e .. -e
".[dev]"` from `server/`), not declared as a `file://../` direct
reference inside `server/pyproject.toml` — confirmed directly that pip
rejects a relative `file://` URI as a "non-local file URI" even though
it resolves to a real local path, so a dependency string isn't a
portable way to express "the sibling directory." `ragcore`'s own
`pyproject.toml` gains zero new dependencies from this either way.

First pass is deliberately **single-tenant**: one `Corpus` instance,
configured at server startup from environment variables, with no
`tenant_id` anywhere in the HTTP surface. This is not an oversight —
Phase 7's `TenantScopedCorpus` enforces a tenant_id *mechanically*, but
multi-tenant HTTP routing needs a real way to know *which* tenant is
calling, and that is exactly the authentication mechanism Phase 7 item
30 (API keys/JWT/RBAC) explicitly defers to this same server layer. An
API that read `tenant_id` from an unauthenticated request header (e.g.
any caller free to set `X-Tenant-ID` to any value) would not be
isolation — it would be the same shape of false security the Tier 2
shared-entity-id risk in `docs/adr/0007` describes, just moved to a new
layer. Multi-tenant routing is explicitly follow-on work, gated on an
auth mechanism actually existing, not shipped ahead of it to look more
complete than it is.

## Consequences

- `ragcore/` stays exactly as pure as `CLAUDE.md` already commits to —
  no server code, no new required dependency, nothing to audit for a
  consumer who never runs the reference server.
- The reference server is real, working code with its own test suite
  (FastAPI's `TestClient`, no network), not a stub — multipart file
  upload for `/ingest` (not just a server-local path, which would defeat
  the point of an HTTP API), JSON request/response models for
  `/retrieve`/`/ask`, `/sources` listing and deletion, `/health`.
- OpenAPI spec generation (also named in Phase 8) is free once this
  exists — FastAPI generates it from the route/model definitions
  automatically; no separate authoring pass is needed.
- Everything else named in Phase 8 (Python/TypeScript SDKs, gRPC, MCP
  server, Docker/K8s manifests, Redis, feature flags, load tests,
  backup/restore runbooks) builds on top of this server existing and is
  tracked as separate, still-open follow-on work in
  `docs/ARCHITECTURE.md`, not attempted in this same pass.
