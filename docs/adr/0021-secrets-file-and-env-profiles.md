# 0021: Mounted secrets file support, and dev/staging/prod env profiles

## Status

Accepted.

## Context

Two remaining pieces of the deployment task's "configuration" section:
"Secrets read from env or a mounted secrets file, never baked into
images or committed," and "Separate dev, staging and prod profiles."

## Decisions

**Secrets file support uses pydantic-settings' own `secrets_dir`,
not a hand-rolled `_FILE`-suffix convention.** Confirmed directly: a
file named after a field (lowercase - e.g. `database_url`) under
`secrets_dir` populates that field, and a real environment variable of
the same name still wins when both are present - exactly "env or a
mounted secrets file," for free, with no new code in `Settings`
itself beyond pointing `secrets_dir` at `/run/secrets` (the standard
Docker/Compose-secrets convention), configurable via
`RAGCORE_SECRETS_DIR`.

**`secrets_dir` is only passed when the directory actually exists.**
pydantic-settings emits a `UserWarning` on every construction
otherwise (confirmed directly) - every local/test run doesn't have
`/run/secrets`, and a warning on every single `Settings()` call in the
test suite would drown out anything that actually matters. Checked
once via a plain `os.environ` read at import time, not a pydantic
field, since `model_config` is fixed at class-definition time and
can't depend on another field's value.

**Three profile files (`server/.env.{dev,staging,prod}.example`),
each short and differential, not three full copies of
`.env.example`.** `.env.example` remains the exhaustive reference for
every variable; each profile file only sets what meaningfully differs
for that environment (log level, which pieces are optional vs.
required, where secrets should come from) - a full copy of
`.env.example` three times over would drift the moment one variable's
default changed and the other two didn't get updated.

**The staging profile defers to `docker-compose.yml`'s own `x-app-env`
anchor for network-topology variables** (`DATABASE_URL`/`REDIS_URL`/
`S3_ENDPOINT_URL`/`S3_BUCKET`/`OTEL_EXPORTER_OTLP_ENDPOINT`), which
Compose already hardcodes to the in-network service names regardless
of what's in `.env` (ADR 0018) - the staging profile only needs to
carry what Compose does *not* already decide: credentials and
auth/tenant configuration.

## Consequences

- The `.gitignore` broad `.env.*` pattern needed two more negations
  (`!.env.*.example`, `!*/.env.*.example`) alongside the existing
  `!.env.example`/`!*/.env.example` - without them these new profile
  files would have been silently excluded from version control.
- Prod's profile is a starter, not a hardened checklist - this task's
  scope is staging; a real prod rollout needs its own review beyond
  what this file documents.
