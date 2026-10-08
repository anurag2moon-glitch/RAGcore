# ragcore reference server

A minimal FastAPI service exposing one `ragcore.Corpus` over HTTP:
ingest a file, retrieve, ask, list/delete sources. Single-tenant and
unauthenticated by default; set `RAGCORE_API_KEYS` to turn on real
multi-tenant HTTP routing (every route scoped through
`ragcore.tenancy.TenantScopedCorpus`) — see
[`docs/adr/0008-api-server-as-sibling-package.md`](../docs/adr/0008-api-server-as-sibling-package.md)
and
[`docs/adr/0012-api-key-auth-and-multi-tenant-routing.md`](../docs/adr/0012-api-key-auth-and-multi-tenant-routing.md)
in the repo root for the full reasoning and what's deliberately still
not here (JWT/OAuth, RBAC, billing-ready token-cost metering).

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ..              # ragcore itself
pip install -e ".[dev]"        # this server + test tooling
```

## Run

```
export RAGCORE_DB_PATH=./data/corpus.lancedb
uvicorn app.main:app --reload
```

### Async ingestion (ADR 0014/0017)

```
export REDIS_URL=redis://localhost:6379/0
export S3_ENDPOINT_URL=http://localhost:9000   # MinIO or any S3-compatible endpoint
export S3_ACCESS_KEY_ID=...
export S3_SECRET_ACCESS_KEY=...
export S3_BUCKET=ragcore

# in one terminal: the API, as above
uvicorn app.main:app --reload

# in another: the worker (the module path must stay importable - see
# app/worker/ingest_job.py's own docstring for why)
rq worker --url "$REDIS_URL" ingest
```

`POST /ingest` returns immediately with `{job_id, source_id, status:
"queued"}` — `GET /jobs/{job_id}` to poll (`status`: `queued` ->
`started` -> `finished` or `failed`; `result` populated once
`finished`, `error` once `failed`). Without `REDIS_URL`/`S3_*` set,
`/ingest` fails fast with `503` rather than silently falling back to
the old synchronous behavior (see ADR 0014's own "Consequences").

### Multi-tenant mode (optional)

```
export RAGCORE_API_KEYS="key-for-tenant-a:tenant-a,key-for-tenant-b:tenant-b"
```

Once set, every request must carry a valid key
(`Authorization: Bearer <key>` or `X-API-Key: <key>`) — `401` if
missing/invalid, `403` if the request tries to touch another tenant's
data (e.g. `DELETE /sources/{id}` for a source you don't own).
`app.auth.InMemoryApiKeyStore` is a reference implementation (no
rotation, no expiry, plaintext comparison) — implement the
`ApiKeyStore` Protocol against your own key-issuance backend for a
real deployment.

### Rate limiting (optional)

```
export RAGCORE_RATE_LIMIT_PER_MINUTE=60
```

Token-bucket per key (per tenant_id in multi-tenant mode, else per
client IP) — a burst up to the configured rate is allowed immediately,
then refills continuously. Exceeding it returns `429`. `/health` is
always exempt. Unset means no rate limiting at all, identical to this
server's behavior before this existed.

### Usage metering

```
GET /usage      # Authorization/X-API-Key required — returns only your own counts
```

Request counts per endpoint for the calling tenant, e.g.
`{"GET /sources": 12, "POST /ingest": 2}`. Scoped to request counts,
not token/cost metering — this server shares one embedder across every
tenant, so attributing its token usage back to individual tenants
isn't well-defined without a bigger redesign; see
`app/usage.py`'s own docstring. `400` in single-tenant mode (no
tenant_id to meter by). `app.usage.InMemoryUsageMeter` is a reference
implementation (per-process, resets on restart) — persist this for a
real deployment.

### Per-tenant quotas (optional)

```
export RAGCORE_TENANT_QUOTAS="tenant-a:1000,tenant-b:5000"
```

A **lifetime total** request cap per tenant (not a recurring
daily/monthly quota — that needs real time-windowing this reference
implementation doesn't build; see `app/quota.py`'s own docstring),
reusing `/usage`'s own counts so the two can never disagree. A tenant
with no entry is unlimited. Exceeding it returns `402`; the rejected
request is never counted (checked before `app.usage.record_usage`
runs). Unset means no quota enforcement at all.

Visit `/docs` for the interactive OpenAPI UI (generated automatically
by FastAPI from `app/main.py`'s routes and `app/models.py`'s Pydantic
models — no separate authoring step).

### Rules (Phase 10, optional)

```
export RAGCORE_RULES_DB_PATH=./data/rules.sqlite3
```

The `/rules/*` routes (`app/rules_routes.py`) need this set, and need
a domain pack registered (empty by default — zero hard dependency on
any specific pack, e.g. Lal Kitab) via
`app.dependency_overrides[get_domain_pack_registry]`; `/rules/extract`
additionally needs `get_rule_extraction_pipeline` overridden with a
real `ClaudeRuleCandidateDetector`/`Extractor`/`Verifier` triple (`503`
until configured). In multi-tenant mode, every route filters to global
rules plus only the calling tenant's own private ones — enforced here,
in the route layer, not inside `RuleStore` itself.

**`/rules/extract`'s own token-cost quota** (separate from the generic
per-request `RAGCORE_TENANT_QUOTAS` above — extraction cost varies
enormously job to job, which request-count quotas can't capture):

```
export RAGCORE_EXTRACTION_TOKEN_QUOTAS="tenant-a:100000,tenant-b:500000"
```

A lifetime total token cap per tenant, metered from the real
`input_tokens + output_tokens` on every Claude API response
(`ClaudeRuleCandidateDetector`/`Extractor`/`Verifier`'s own
`.usage: ragcore.embedding.cost.UsageTracker`), not a client-side
estimate. Checked *before* a job runs (rejects `402` if already over
budget from past jobs) and recorded *after* it returns (the job's own
cost isn't known until then) — a single job can therefore push a
tenant over quota by however much that job costs; only the *next* one
is rejected. A tenant with no entry is unlimited; unset means no
enforcement.

**Known limitation**: `/rules/extract` reads chunks via
`corpus.store.get_by_source()` directly, same as every `build_*()`
pipeline in `ragcore` (graph/concepts/claims/contextual embeddings) —
none of them go through `TenantScopedCorpus`'s own wrapped methods, so
in multi-tenant mode extraction currently reads across **every**
tenant's ingested chunks, not just the calling tenant's. This is a
pre-existing, systemic gap in `ragcore`'s pipeline builders, not
something specific to rule extraction — see
`ragcore.tenancy.TenantScopedCorpus`'s own docstring for the other
store/retriever paths it similarly doesn't cover yet.

## Test

```
pytest -q
```

No network, no real Voyage/LanceDB credentials needed —
`app.corpus_provider.get_corpus` and `app.auth.get_api_key_store` are
overridden with in-memory test doubles via FastAPI's
`app.dependency_overrides` (`test_api.py` for the single-tenant path,
`test_auth.py` for the multi-tenant one).

## Endpoints

| Method | Path | Body | Notes |
|---|---|---|---|
| `GET` | `/health`, `/health/live` | — | liveness check (process up, no I/O), never requires a key |
| `GET` | `/health/ready` | — | readiness: `503` if any *configured* dependency (Postgres/Redis/the vector store) is unreachable; an unconfigured one never counts as a failure |
| `POST` | `/ingest` | multipart `file` | **async as of ADR 0014/0017** — uploads to object storage and enqueues a job, returns `202 {job_id, source_id, status}`; `503` if `REDIS_URL`/`S3_*` aren't configured |
| `GET` | `/jobs/{job_id}` | — | poll for ingestion job status/result; `404` unknown job_id, `503` if `REDIS_URL` isn't configured |
| `POST` | `/retrieve` | `{query, k, filters, min_score}` | |
| `POST` | `/ask` | `{query, k, filters}` | `400` if the corpus has no `Generator` configured |
| `GET` | `/sources` | — | |
| `DELETE` | `/sources/{source_id}` | — | `403` in multi-tenant mode if the source belongs to another tenant |
| `GET` | `/usage` | — | your own request counts per endpoint; `400` in single-tenant mode |
| `GET` | `/rules` | — | `?domain=...&book_id=...` |
| `POST` | `/rules/match` | `{domain, facts, book_id, include_unverified}` | |
| `POST` | `/rules/{rule_id}/approve` | `{reviewer, reason}` | mutates `review_status` in place |
| `POST` | `/rules/{rule_id}/reject` | `{reviewer, reason}` | reason required |
| `POST` | `/rules/{rule_id}/edit` | `{reviewer, reason, effect, confidence, remedy}` | creates a new version |
| `POST` | `/rules/extract` | — | `?domain=...&source_id=...`; `404` with no domain pack registered, `503` with no extraction pipeline configured |

Every route except `/health` returns `401` in multi-tenant mode if the
request has no valid API key, and (except `/usage` itself) `429` if
`RAGCORE_RATE_LIMIT_PER_MINUTE` is set and exceeded, or `402` if
`RAGCORE_TENANT_QUOTAS` is set and this tenant's lifetime quota is
used up.
