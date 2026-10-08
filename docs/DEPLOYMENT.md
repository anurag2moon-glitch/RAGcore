# Deployment: minimal staging (single node)

Covers the Docker/Compose deployment of `server/` — one API container,
one ingestion worker, Postgres (tenant/API-key registry), Redis (the
async ingest job queue), MinIO (S3-compatible object storage), and a
LanceDB data volume. See [ADR 0014](adr/0014-minimal-staging-deployment-architecture.md)
for why this topology was chosen, and [ADR 0018](adr/0018-containers-and-compose-stack.md)
through [0022](adr/0022-ci-type-check-build-images-integration-test.md)
for every decision made building it. **No Kubernetes or autoscaling in
this setup** — single node, `docker compose`'s own `restart:
unless-stopped` is the full extent of its self-healing. A Kubernetes
migration is a future ADR, not something this document covers.

## Setup

Prerequisites: Docker and Docker Compose. Nothing else — `make up`
builds every image itself.

```bash
cp server/.env.staging.example server/.env
# edit server/.env: at minimum set VOYAGE_API_KEY and ANTHROPIC_API_KEY
# (ingestion/retrieval/generation all need Voyage; /ask needs Anthropic too)

make up        # docker compose up -d --build
make logs      # follow every container's logs (ctrl-C to stop watching, doesn't stop the stack)
```

`make up` brings up, in dependency order: `postgres`, `redis`, `minio`
→ `minio-init` (creates the bucket, since a fresh MinIO has none) and
`migrate` (one-shot, runs `alembic upgrade head`) → `api` and `worker`
(only once `migrate` exits `0`) → `otel-collector`. Check everything
is actually healthy:

```bash
curl http://localhost:8000/health/ready
# {"status": "ready", "checks": {"corpus": "ok", "postgres": "ok", "redis": "ok"}}
```

A `503` here with a per-check error message is the first place to
look when something's wrong — see **Troubleshooting** below.

### Seed a tenant and API key

`docker-compose.yml` sets `DATABASE_URL` unconditionally, which
activates `PostgresApiKeyStore`-backed multi-tenant auth regardless of
whether `RAGCORE_API_KEYS` is set ([ADR 0016](adr/0016-postgres-tenant-registry-and-migrations.md)) —
every route except `/health*` and `/metrics` needs a valid `X-API-Key`
once the stack is up. Seed one:

```bash
make seed       # tenant "staging", a freshly-generated key printed to stdout
# or: docker compose exec api python scripts/seed_staging.py my-tenant my-api-key
```

### Try it

```bash
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: <your key>" \
  -F "file=@some_book.pdf"
# {"job_id": "...", "source_id": "...", "status": "queued"}

curl http://localhost:8000/jobs/<job_id> -H "X-API-Key: <your key>"
# poll until "status": "finished"

curl -X POST http://localhost:8000/ask \
  -H "X-API-Key: <your key>" -H "Content-Type: application/json" \
  -d '{"query": "your question", "k": 5}'
```

### Shutting down

```bash
make down       # docker compose down - containers stop, volumes persist
docker compose down -v   # also deletes postgres_data/minio_data/lancedb_data - destructive
```

## Configuration reference

Every variable is documented in `server/.env.example` (the exhaustive
reference) and `app/settings.py` (`Settings`, the validated model that
reads them — fails fast on anything missing or malformed at process
startup, before the port ever binds). Three starter profiles exist,
each only setting what meaningfully differs for that environment —
see [ADR 0021](adr/0021-secrets-file-and-env-profiles.md):

| Profile | File | Notes |
|---|---|---|
| dev | `server/.env.dev.example` | Local, single-tenant, synchronous — no Postgres/Redis/MinIO required unless you uncomment them. |
| staging | `server/.env.staging.example` | Matches `docker-compose.yml`'s own expectations — Compose overrides the network-topology variables (`DATABASE_URL`/`REDIS_URL`/`S3_ENDPOINT_URL`/`S3_BUCKET`/`OTEL_EXPORTER_OTLP_ENDPOINT`) regardless of what's in `.env`, so this file only needs credentials and auth config. |
| prod | `server/.env.prod.example` | Starter, not a hardened checklist — this task's scope is staging. Expects secrets via a mounted file, not inline. |

### Secrets

Two ways to supply a secret, and either works for any `Settings`
field: a real environment variable, or a file named after the field
(lowercase — e.g. `database_url`, `s3_secret_access_key`) under
`RAGCORE_SECRETS_DIR` (default `/run/secrets`, the standard
Docker/Compose-secrets convention). A real environment variable always
wins when both are present. Never commit a real `.env`, and never bake
a secret into an image layer.

### Tenant data paths, object storage keys, table names

All three come from `app/naming.py`, not hardcoded at each call site —
`tenant_data_path()`, `raw_file_object_key()`, `backup_object_key()`,
plus the two Postgres table name constants. Change how tenant
isolation works (e.g. per-tenant buckets, a different table-naming
scheme) by changing this one module; nothing else in `server/`
hardcodes any of these.

## Operations

- **Logs**: `make logs`, or `docker compose logs -f <service>` for one
  container. Every log line is one JSON object (`app/logging_config.py`)
  tagged with `request_id` — an auto-generated UUID per request, or
  whatever `X-Request-ID` the caller sent, round-tripped onto the
  response header too.
- **Metrics**: `curl http://localhost:8000/metrics` — Prometheus text
  exposition of per-route request counts/latency. Separate from the
  OTLP metrics `fastapi[opentelemetry]` already exports to the
  collector (see below) — this is the plain-text endpoint a Prometheus
  server or a bare `curl` scrapes directly.
- **Tracing**: every request's spans export via OTLP to the
  `otel-collector` service (`server/otel-collector-config.yaml`),
  which currently logs them to its own stdout (`docker compose logs
  otel-collector`) — swap its `exporters`/`service.pipelines` for a
  real backend (Jaeger/Tempo/...) without touching `api`/`worker` at
  all, which only ever know about the collector's own OTLP endpoint.
- **Graceful worker shutdown**: needs no manual intervention.
  `docker compose stop worker` (or `down`) sends `SIGTERM`, which
  `rq`'s own `Worker.request_stop` turns into "finish the in-flight
  ingestion job, then exit" — a second `SIGTERM`/the compose
  `stop_grace_period` timeout force-exits instead.
- **Re-running migrations**: `make migrate` (safe to run any time —
  Alembic no-ops once the schema is current).

## Backup and restore

`make backup TAG=<name>` / `make restore TAG=<name>` — see
[ADR 0020](adr/0020-backup-restore.md) for the full design. All three
pieces (the Postgres tenant/API-key registry, the LanceDB + rules
SQLite + tenant-data volume, and the raw-files object storage bucket)
land in the **same** MinIO bucket under a `backups/<tag>/` prefix, not
a separate host directory — the scripts need only what the `api`
container already has (network access to Postgres/S3, the mounted
data volume), no extra bind mount.

```bash
make backup TAG=2026-10-08
make restore TAG=2026-10-08
```

Restoring **replaces** current state with the backup's state — it
does not merge. Postgres restore `TRUNCATE`s each table before
reloading; the data-dir and raw-files restores overwrite whatever's
currently there.

**This mechanism was verified with a real round-trip** (not just
written and reviewed): `server/tests/test_backup_restore.py` runs the
data-dir and object-storage halves against a moto-mocked bucket and
real temp directories, simulating total data loss between backup and
restore and asserting the original bytes come back. The Postgres half
is `pytest -m postgres`-gated (no live Postgres in the development
environment this was built in) — if you want to verify it yourself
against a real server:

```bash
cd server
RAGCORE_TEST_POSTGRES_URL=postgresql://... pytest -m postgres tests/test_backup_restore.py -v
```

**This is not a substitute for taking the MinIO volume itself
off-node.** A real disaster that destroys the whole `minio_data`
volume takes the backups with it, since they live in the same bucket
they're backing up pieces of. For actual staging use, mirror the
bucket elsewhere (e.g. `mc mirror`) — orthogonal to, and not handled
by, this mechanism.

The library-level ingestion-versioning mechanism
(`ragcore.versioning.snapshot`, documented in
[docs/RUNBOOK.md](RUNBOOK.md)) is a **different, complementary**
thing — it rolls back a *bad ingest* using LanceDB's own tag/restore
API plus SQLite online backups, called from inside a Python process
that already has the relevant stores open. It is not wired into
`server/`'s deployment scripts; nothing here calls it.

## Troubleshooting

**`/health/ready` returns 503.** Read the `checks` object in the
response body — only *configured* dependencies are checked, and each
one's own error message is whatever the real connection attempt
failed with (e.g. `error: connection refused` for Postgres/Redis). Fix
the specific dependency named, not the others.

**`POST /ingest` returns 503.** `REDIS_URL`/`S3_*` aren't configured —
check `docker compose exec api env | grep -E 'REDIS_URL|S3_'`. In the
Compose stack these are always set by `docker-compose.yml`'s own
`x-app-env` anchor; a 503 here means the `api` container isn't picking
up the environment you expect, not that the stack itself is
misconfigured.

**`GET /jobs/{job_id}` never reaches `"finished"`.** Check the worker
is actually running and connected: `docker compose logs worker`. A
job stuck `"queued"` usually means the worker container exited or
never started — `docker compose ps` to check its status, `docker
compose up -d worker` to restart it.

**Every route returns `401`.** Multi-tenant auth is active
(`DATABASE_URL` is set unconditionally in Compose) and you're missing
`X-API-Key`, or using a key that was never seeded — run `make seed`
and use the key it prints, or re-run
`scripts/seed_staging.py <tenant> <key>` with a key you choose.

**`make backup`/`make restore` fails with a connection error.** Both
need to run inside the `api` container (`docker compose exec api
python scripts/backup.py <tag>`), which already has network access to
Postgres/MinIO — running them directly on the host instead will fail
to resolve `postgres`/`minio` as hostnames (those only resolve inside
the Compose network).

**`alembic upgrade head` fails / `migrate` never completes
successfully.** `api`/`worker` won't start until it does
(`service_completed_successfully`) — `docker compose logs migrate` for
the real error; almost always either Postgres isn't healthy yet
(`docker compose ps postgres` — should show `healthy`, not
`starting`) or `DATABASE_URL`'s credentials don't match
`POSTGRES_USER`/`POSTGRES_PASSWORD` in `docker-compose.yml`.

**A Docker image build fails.** `server/requirements.lock.txt` is a
pinned snapshot, not pyproject.toml's own `>=` floors — if a
dependency was added/changed in `server/pyproject.toml` without
regenerating the lockfile, the build installs stale pins. Regenerate
per the lockfile's own header comment, then rebuild.
