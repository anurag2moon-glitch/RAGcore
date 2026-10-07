# 15. A single validated settings model for `server/`, fail-fast at import time

Date: 2026-10-07

## Status

Accepted

## Context

Every provider module in `server/app/` (`corpus_provider.py`,
`auth.py`, `rate_limit.py`, `quota.py`, `rules_provider.py`,
`extraction_quota.py`) read its own environment variable directly via
`os.environ.get(...)`, behind an `@lru_cache`'d singleton constructed
lazily on first use. This meant a misconfigured or malformed value
(e.g. `RAGCORE_API_KEYS=no-colon-here`) only failed the first time a
request actually touched that provider — `RAGCORE_DB_PATH` being
unset, for instance, only surfaced as an error on the first `/ingest`
or `/retrieve` call, not at startup. `RAGCORE_TENANT_QUOTAS` and
`RAGCORE_EXTRACTION_TOKEN_QUOTAS` had also independently grown
byte-for-byte duplicate `"tenant:limit"` string parsers.

ADR 0014's deployment work adds several more environment variables
(`DATABASE_URL`, `REDIS_URL`, `S3_*`) with their own ways to be
malformed (wrong URL scheme, missing credentials), making "which
config is broken" harder to diagnose after the fact if each one keeps
failing lazily and independently.

## Decision

One `pydantic_settings.BaseSettings` model, `app.settings.Settings`,
covering every environment variable this server reads — the existing
ones (same exact names, so nothing about how this server is
configured from the outside changes) plus every new one ADR 0014
needs. `get_settings()` is called once, at `app.main` **module import
time** (before `FastAPI()` is even constructed) — not lazily inside a
request handler — so a bad environment fails the process before
`uvicorn` ever binds a port, with one structured `pydantic.ValidationError`
naming every broken field at once, not a chain of individually-discovered
runtime errors.

Three provider modules' independently-duplicated `"key:value,..."`
parsers (`auth._parse_api_keys_env`, `quota._parse_quotas_env`,
`extraction_quota._parse_quotas_env` — the latter two byte-for-byte
identical) are consolidated into `app.env_parsing.parse_colon_pairs`/
`parse_int_quotas`, used both by each provider (unchanged behavior)
and by `Settings`' own `@field_validator`s (so a malformed value is
caught at startup, in the exact same parsing logic that will later
act on it — not a second, independently-written validation rule that
could drift from the real parser).

Every field stays optional with a safe default — `Settings()` never
fails on a *missing* variable, matching every existing opt-in piece of
this server (`RAGCORE_API_KEYS` unset = no auth, etc.); it only fails
on a *malformed* one. This is what keeps every existing test passing
unchanged: the test suite sets no environment variables at all (every
test configures behavior via `app.dependency_overrides`), so
`get_settings()` succeeds trivially at `app.main` import time in every
test process.

## Consequences

- A genuinely new dependency for `server/` only: `pydantic-settings`
  (not `ragcore/` — `ragcore` itself still reads zero environment
  variables, per `docs/CONFIGURATION.md` and ADR 0009).
- `server/.env.example` documents every variable `Settings` reads, one
  file, always in sync with the model by construction (anyone adding a
  field to `Settings` should add the matching line here — not
  mechanically enforced, but it's the one place to check).
- `mcp-server/` is **not** touched — it has its own, separate,
  already-duplicated `_corpus_singleton`/`os.environ.get` pattern
  (`mcp_server/corpus_provider.py`'s own docstring: "Deliberately
  duplicates server/'s... shape rather than importing it," since the
  two are independently deployable artifacts). Bringing it onto the
  same settings model is a separate, smaller follow-on, not bundled
  into this ADR.
- The "single helper module" the deployment request asks for, for
  "tenant data paths, object storage prefixes and table naming," is a
  further module (`app.naming`, next unit) that reads from this
  `Settings` object rather than hardcoding any of those strings —
  this ADR is the prerequisite for that one, not that module itself.
