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
not here (JWT/OAuth, RBAC, usage metering).

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

Every route except `/health` returns `401` in multi-tenant mode if the
request has no valid API key, and `429` if `RAGCORE_RATE_LIMIT_PER_MINUTE`
is set and exceeded.
