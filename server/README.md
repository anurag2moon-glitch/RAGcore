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
| `GET` | `/health` | — | liveness check, never requires a key |
| `POST` | `/ingest` | multipart `file` | dispatches by the uploaded filename's extension; tags every chunk with the resolved tenant_id in multi-tenant mode |
| `POST` | `/retrieve` | `{query, k, filters, min_score}` | |
| `POST` | `/ask` | `{query, k, filters}` | `400` if the corpus has no `Generator` configured |
| `GET` | `/sources` | — | |
| `DELETE` | `/sources/{source_id}` | — | `403` in multi-tenant mode if the source belongs to another tenant |
| `GET` | `/usage` | — | your own request counts per endpoint; `400` in single-tenant mode |

Every route except `/health` returns `401` in multi-tenant mode if the
request has no valid API key, and (except `/usage` itself) `429` if
`RAGCORE_RATE_LIMIT_PER_MINUTE` is set and exceeded, or `402` if
`RAGCORE_TENANT_QUOTAS` is set and this tenant's lifetime quota is
used up.
