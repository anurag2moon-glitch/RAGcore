# 14. Minimal staging deployment architecture

Date: 2026-10-07

## Status

Accepted

## Context

`docs/adr/0008-api-server-as-sibling-package.md` named "Docker/K8s
manifests, Redis, feature flags, load tests, backup/restore runbooks"
as explicit follow-on work, deferred until a real deployment was
actually being built. That request has now arrived, scoped
deliberately small: a single-node, production-*shaped* staging
deployment — containers, config, migrations, basic observability,
backup/restore, CI — explicitly **not** Kubernetes/autoscaling (a
later ADR, per the request's own rules).

`server/` (the reference FastAPI app) and `mcp-server/` already exist
as sibling packages per ADR 0008. Today, `/ingest` runs synchronously
inside the request — a request blocks until `Corpus.ingest()` returns.
The request explicitly asks for a separate `api` and `worker`
container in Compose, and for "workers finish or checkpoint the
current ingestion stage before exiting" — language that only makes
sense if ingestion is an asynchronous job a worker process pulls and
processes, not something run inline in a web request. It also asks
for Postgres, Redis, and MinIO (S3-compatible storage) as first-class
Compose services, and for "tenant data paths, object storage prefixes
and table naming" to come from config, not be hardcoded.

Three pieces of `ragcore` already exist that this can build on rather
than duplicate:
- `ragcore.raw_files.S3RawFileStore` — already works against any
  S3-compatible endpoint (including MinIO, via `endpoint_url`),
  already the right abstraction for "durable storage for the original
  file bytes" per its own module docstring.
- `ragcore.ingestion.PostgresIngestCheckpointStore` — proves Postgres
  is already a supported backend for *ingestion* checkpointing
  specifically, via the `postgres` extra (`psycopg[binary]`).
- `server/app/auth.ApiKeyStore` — already a swappable Protocol
  (`resolve_tenant(api_key) -> str | None`); `InMemoryApiKeyStore` is
  the only implementation so far, built explicitly to be replaced by
  "a real deployment managing its own key lifecycle... against its own
  backing store" (its own docstring).

## Decision

**Container topology** (all in one `docker-compose.yml`, one `up`):
`api` (FastAPI, `server/`), `worker` (ingestion job consumer),
`migrate` (one-shot Alembic runner, exits after migrating), `postgres`,
`redis`, `minio`, plus named volumes for LanceDB data, the rules
SQLite file, Postgres data, and MinIO data.

**Ingestion becomes an async job, queued through Redis via RQ
(`rq`).** `POST /ingest` now: uploads the file to MinIO via
`S3RawFileStore` (object key = a generated `source_id`-shaped key, the
same abstraction already built for exactly this), enqueues one RQ job
referencing that object, and returns `202` with a `job_id` immediately
— it does not run `Corpus.ingest()` inline anymore. A new
`GET /jobs/{job_id}` reads the job's status/result straight from RQ's
own Redis-backed `Job` object (`rq.job.Job.fetch`), no separate job-
status store needed — RQ already persists exactly that. The `worker`
container runs `rq worker` against the same queue: on pulling a job it
downloads the file from MinIO and calls `Corpus.ingest()` unchanged
(ingestion's own checkpointing, `ragcore.ingestion`, is untouched by
any of this). This was chosen over a hand-rolled queue specifically
because RQ's `Worker` already does the one thing the request asks for
by name — confirmed directly from its source
(`Worker.request_stop`/`handle_warm_shutdown_request`): on the first
SIGTERM it lets the in-flight job finish before exiting ("warm
shutdown"), and only force-exits on a second signal. Docker's own stop
sequence (SIGTERM, then SIGKILL after a grace period) maps onto this
exactly, with zero custom signal-handling code needed.

**Postgres holds the tenant/API-key registry, not ragcore's own
retrieval data.** A new `PostgresApiKeyStore` (`server/app/`, not
`ragcore/` — tenant/key *management* is this HTTP server's own
concern, same boundary ADR 0008 already draws) implements the
existing `ApiKeyStore` Protocol unchanged. Alembic migrates its
schema (`tenants`, `api_keys` tables). LanceDB/SQLite rule data stay
exactly where they are today (file-based, on a Compose volume) — this
does **not** migrate ragcore's own storage to Postgres; that was never
asked for and would be a much larger, separate decision.

**MinIO is the one object store, used for two distinct things**: raw
ingested files (already via `S3RawFileStore`, now also the ingest-job
handoff mechanism between `api` and `worker`) and, later, backup
archives (see the backup/restore ADR). One bucket naming/prefix scheme
for both, from config (see the settings ADR) — not hardcoded, per the
request's own explicit rule.

**No Kubernetes, no autoscaling, no Helm charts in this pass** — the
request's own rule. Mentioned once, here, as explicit future work: a
K8s manifest set would replace `docker-compose.yml`'s service
definitions with Deployments/Services/a Job for migrations, and would
need its own ADR for the harder parts Compose doesn't have an opinion
on (HPA metrics for the worker specifically, a shared `ReadWriteMany`
volume or a move of LanceDB data to a networked backend, since a
single-node hostPath volume doesn't survive a pod rescheduling to a
different node).

## Consequences

- `/ingest`'s response contract changes (`200` + full `IngestResponse`
  body -> `202` + `{job_id}`) — a breaking change to that one endpoint
  for any existing caller, including both SDKs, which need a matching
  `ingest_async`/job-polling method (or `ingest()` itself becomes
  poll-until-done sugar over the two calls — a per-SDK decision made
  at implementation time, not re-litigated here).
- `rq` becomes a new dependency of `server/` (not `ragcore/` — the
  library itself still has no job-queue opinion, consistent with ADR
  0008/0009). `psycopg[binary]` and `alembic` likewise become
  `server/`-only dependencies, reusing `ragcore[postgres]`'s own choice
  of driver rather than introducing a second one.
- Every existing environment variable this server reads
  (`RAGCORE_DB_PATH`, `RAGCORE_API_KEYS`, `RAGCORE_RATE_LIMIT_PER_MINUTE`,
  `RAGCORE_TENANT_QUOTAS`, `RAGCORE_RULES_DB_PATH`,
  `RAGCORE_EXTRACTION_TOKEN_QUOTAS`) moves into the new validated
  settings model (separate ADR) rather than staying as scattered
  `os.environ.get()` calls — this pass touches every existing provider
  module, not just the new ones.
- `mcp-server/`, `sdk/python/`, `sdk/typescript/` are **not** touched
  by the async-ingest change in this pass — they still call the
  synchronous `Corpus.ingest()` path directly (`mcp-server/`) or the
  old `/ingest` contract (both SDKs), which now breaks for the SDKs
  specifically. Fixing the SDKs to match the new `/ingest` contract is
  in-scope follow-on work tracked alongside this ADR's implementation,
  not deferred indefinitely.
- Local dev without Docker keeps working for everything *except*
  `/ingest`: `PostgresApiKeyStore`/auth/rate-limiting/quotas stay
  opt-in via settings exactly as today, but `/ingest` itself now
  requires Redis unconditionally once this lands — maintaining both a
  synchronous fallback path and the async job path would mean two
  ingestion code paths with two sets of edge cases (and two things to
  keep correct under the checkpointing/dead-letter semantics
  `ragcore.ingestion` already provides once, today), for a feature
  (local dev with zero infra) the `.env` dev profile already covers
  by pointing `REDIS_URL` at a real local Redis instead. `GET /health`
  stays a pure liveness check with no Redis dependency, so the service
  itself still starts and reports healthy without it; only `/ingest`
  calls fail fast with a clear "Redis not configured" error.
