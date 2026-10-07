# 16. Postgres tenant/API-key registry, managed by Alembic

Date: 2026-10-07

## Status

Accepted

## Context

ADR 0014 decided Postgres holds the new tenant/API-key registry, not
ragcore's own retrieval data. `app.auth.ApiKeyStore` is already a
swappable Protocol with exactly one implementation
(`InMemoryApiKeyStore`), built explicitly to be replaced — its own
docstring already named this. The minimal-staging-deployment request
asks for Alembic migrations run automatically on startup via a
one-shot migration container, and a seed script for a staging tenant
and API key.

## Decision

`PostgresApiKeyStore` (`server/app/postgres_auth.py`) implements
`ApiKeyStore` unchanged (`resolve_tenant`), plus admin operations
(`create_tenant`/`create_api_key`/`revoke_api_key`) the seed script
and any future tenant-management surface call. Table/column shape and
psycopg conventions mirror `ragcore.ingestion.PostgresIngestCheckpointStore`
exactly (`%s` placeholders, `TIMESTAMPTZ`, `connection.transaction()`,
`ON CONFLICT ... DO UPDATE SET` upserts) — reusing a pattern already
reviewed once rather than inventing a second style.

**Alembic manages the schema, raw SQL only, no ORM models** —
consistent with every other store in this codebase family, none of
which use an ORM. `alembic/env.py` reads `DATABASE_URL` from
`app.settings.get_settings()` (not a second, separately-configured
`sqlalchemy.url` in `alembic.ini`) and connects via SQLAlchemy's
`postgresql+psycopg` dialect — SQLAlchemy 2.x's native psycopg **3**
support, matching the same driver `ragcore[postgres]`/this server's
own `psycopg[binary]` dependency already chose, not a second Postgres
driver (`psycopg2`) pulled in just for Alembic. SQLAlchemy itself is
Alembic's own required dependency here (Alembic cannot run without
it) — it is not used anywhere else in this server; no ORM models, no
`Session`, no second data-access layer.

Migration files use **literal** table names (`tenants`, `api_keys`),
not `app.naming`'s constants — a migration is a frozen historical
record of what was actually run, and importing a live application
constant into it would mean the file's behavior could silently change
if that constant were later edited, which defeats the point of a
migration. `app.naming.TENANTS_TABLE`/`API_KEYS_TABLE` exist so
`PostgresApiKeyStore` and the seed script can't drift from *each
other*; the migration is the one place those names originate, by
design.

`DATABASE_URL` takes precedence over `RAGCORE_API_KEYS` in
`app.auth.get_api_key_store()` when both happen to be set — Postgres
is the real, persistent store; the env-var-defined key list exists
purely for local dev without a Postgres container running.

**Tested only against a real Postgres, opt-in** — same disclaimer as
`PostgresIngestCheckpointStore`'s own module docstring, for the same
reason (no embedded Postgres exists to test against unconditionally
the way SQLite's stdlib module allows, and no Docker is available in
the environment this was written in to spin up a real one).
`server/tests/test_postgres_auth.py` is gated behind
`pytest -m postgres` + `RAGCORE_TEST_POSTGRES_URL`, skips cleanly
otherwise, mirroring `tests/test_postgres_ingest_checkpoint_store.py`'s
exact convention. Written and reviewed against documented psycopg
3/Alembic/SQLAlchemy behavior — confirmed directly (not assumed) that
`env.py` correctly builds the dialect-qualified connection string and
reaches the actual TCP connection attempt before failing, by running
`alembic current` against an unreachable port and inspecting that the
only failure is a real `ConnectionRefused`, not a configuration error
upstream of it.

## Consequences

- `server/` gains `psycopg[binary]`, `alembic`, and (transitively)
  `sqlalchemy` as real, always-installed dependencies — not optional
  extras, unlike `ragcore[postgres]`'s own opt-in shape, since this
  server's tenant/key registry is meant to work the same way in every
  environment that runs it (even local dev without a Postgres
  container just gets `get_api_key_store() -> None`, same as today).
- The `migrate` Compose service (`alembic upgrade head`, next unit)
  must complete before `api`/`worker` start, or `PostgresApiKeyStore`
  will raise against tables that don't exist yet — this is an
  ordering dependency Compose's own `depends_on`/healthcheck
  mechanism expresses, not something this module tries to detect or
  retry around.
- A tenant/key *management* HTTP surface (create/list/revoke via an
  authenticated admin endpoint, rather than only the seed script) is
  explicit future work, not attempted here — this pass only needed
  "persistent store exists, seed script works."
