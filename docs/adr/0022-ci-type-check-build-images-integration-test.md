# 0022: CI type-check gate, image builds, and a real Compose integration test

## Status

Accepted.

## Context

The deployment task's CI section asked for lint, type-check, unit
tests, build images, and an integration test against the Compose
stack (ingest a small sample book, run a search, check the citation).
Lint and unit tests already existed in `.github/workflows/ci.yml`
(report-only lint, a real `pytest -q` gate per job). This covers the
remaining three.

## Decisions

**Type-check is a blocking gate for `server/`, not report-only like
the pre-existing repo-wide lint step.** Ran `mypy app scripts` before
deciding either way: zero findings across 26 source files once two
small real issues were fixed (an optional-import pattern needing the
same `# type: ignore[assignment]` `ragcore/raw_files/s3.py` already
uses for its own boto3 import; a `Corpus | TenantScopedCorpus` type
alias needing `TypeAlias` so mypy recognizes it as one rather than a
plain variable). With no backlog to excuse deferring it, report-only
would just be accumulating the same kind of debt the lint step already
carries (141+ findings, explicitly flagged there as "not this pass's
backlog to clean up"). `server/pyproject.toml`'s new `[tool.mypy]`
section scopes this to `app`/`scripts` (not `ragcore/` itself, which
has never been type-checked and is out of scope for this deployment
task) and sets `explicit_package_bases = true`, since `scripts/` has
no `__init__.py` (a namespace package, matching how it's already
imported elsewhere - `tests/test_backup_restore.py`'s `from scripts
import backup, restore`) and mypy otherwise refuses to check it
alongside `app/`.

**"Build images" builds all three `server/Dockerfile` targets
(api/worker/migrate) directly via `docker build --target`,** mirroring
the exact commands in that Dockerfile's own header comment. This is
also the Dockerfile's first real verification end to end - no Docker
daemon exists in the environment it was written in (ADR 0018 says so
plainly), so this job failing on a real GitHub Actions runner (which
does have Docker) is the first actual build, not a repeat of something
already confirmed.

**The integration test needs a working embedder+generator, but no
VOYAGE_API_KEY/ANTHROPIC_API_KEY secret is assumed to exist in CI** -
matching this repo's own existing `pytest -m live` exclusion for
exactly that reason. `app/ci_fakes.py` adds a deterministic, offline
`HashingEmbedder` (a bag-of-words hashing-trick embedding - confirmed
directly: two texts sharing vocabulary score meaningfully above zero
cosine similarity, two sharing none score exactly zero) and an
`EchoCitingGenerator` (cites every retrieved chunk verbatim, no LLM
call). Gated behind `Settings.ragcore_use_fake_models`
(`RAGCORE_USE_FAKE_MODELS=1`) - set only in the CI job's own `.env`,
never in a real deployment profile. This is a stronger fake than
`tests/fakes.FakeEmbedder` (every text embeds to the same constant
vector, fine for in-process tests that never check *which* chunk came
back) - a real end-to-end "ingest a book, search it, check the
citation" test needs retrieval to actually discriminate by content, or
the search/citation assertions would be checking nothing.

**The integration test script (`scripts/ci_integration_test.py`) talks
HTTP to the real running stack**, not `docker compose exec` into
Python internals - it's exercising the actual deployed contract
(ingest -> 202 + job_id, poll `/jobs/{id}`, `/retrieve`, `/ask`) the
same way a real client would, which is the point of an integration
test as opposed to another unit test. It seeds its own tenant/key via
`scripts/seed_staging.py` (already existed, ADR 0016) rather than
assuming single-tenant mode, since `docker-compose.yml`'s `DATABASE_URL`
being set unconditionally activates `PostgresApiKeyStore`-backed
multi-tenant auth (`app/auth.py`'s own precedence rule) regardless of
whether `RAGCORE_API_KEYS` is set.

## Consequences

- `RAGCORE_USE_FAKE_MODELS` is a real, if narrow, branch in production
  code (`app/corpus_provider.py`) that only ever matters for this one
  CI job - a deliberate, documented exception to "no placeholder
  code," since the alternative (a real Voyage/Anthropic API key
  committed as a CI secret) would mean every PR run spends real money
  against a real third-party API merely to prove the deployment
  mechanics work, which is a worse trade for this task's purpose.
- The integration test job depends on `build-images` but rebuilds
  the same images again itself via `docker compose up --build` -
  Compose's own build cache makes the second build fast, not free;
  worth revisiting (e.g. `docker compose up -d --no-build` reusing
  images tagged by the first job) if this job's runtime becomes a
  problem.
