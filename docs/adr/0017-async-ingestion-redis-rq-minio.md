# 17. Async ingestion via Redis/RQ, object storage handoff via MinIO

Date: 2026-10-08

## Status

Accepted

## Context

ADR 0014 decided `/ingest` becomes an async job rather than running
`Corpus.ingest()` inline in the HTTP request, queued through Redis via
RQ, with MinIO as the handoff mechanism between the `api` container
(receives the upload) and the `worker` container (actually ingests
it) — reusing `ragcore.raw_files.S3RawFileStore` rather than a shared
volume, so the design works the same whether `api`/`worker` share a
node or not.

## Decision

**Content-addressed handoff, no job payload carrying the file
itself.** `POST /ingest` reads the uploaded bytes, computes
`sha256(bytes)` (the exact scheme every `ragcore` `Loader`
independently re-derives as `Document.source_id` — confirmed by
reading `ragcore/loaders/*.py`'s own `_sha256_file` helpers, not
assumed), uploads under that hash via `S3RawFileStore.put()`, and
enqueues `app.worker.ingest_job.run_ingest_job(source_id=..., tenant_id=...)`.
The job downloads by `source_id` (recovering the original filename
from S3 object metadata via `S3RawFileStore.original_filename()`,
already built for exactly this), stages it under
`app.naming.tenant_data_path(tenant_id) / "incoming"`, and calls
**unmodified** `Corpus.ingest(path, tenant_id=tenant_id)` — no
`TenantScopedCorpus` wrapping needed in the worker at all, unlike
`app.rules_routes.extract_rules`'s own systemic-gap workaround
(`docs/ARCHITECTURE.md` "What's fragile" #11), because `ingest()`
already has first-class `tenant_id=...` write-time tagging.

**`GET /jobs/{job_id}` reads straight from RQ's own `Job` object** — no
separate job-status store, since RQ already persists exactly that in
Redis. Returns `404` for an unknown job_id, `503` if `REDIS_URL` isn't
configured at all (a distinct failure mode from "no such job," kept
distinguishable rather than collapsed into one response).

**No synchronous fallback when Redis/object storage aren't
configured** — `/ingest` returns a clear `503` instead. ADR 0014
already named the reasoning; this ADR is where it's actually enforced
in code, not just stated as intent.

## Consequences

- **A real, executable test suite for this flow exists** —
  `fakeredis` (a real in-Redis-protocol in-memory server, not a stub)
  plus `moto`'s S3 mock let `server/tests/test_worker_ingest_job.py`
  and the updated `test_api.py`/`test_auth.py` exercise the genuine
  enqueue -> `SimpleWorker` burst -> `Job.fetch` round trip, including
  a real RQ serialization constraint caught directly (a job function
  defined inline in a throwaway script fails immediately: "Functions
  from the __main__ module cannot be processed by workers" — this is
  why `run_ingest_job` lives in its own importable module). This is
  meaningfully more verified than ADR 0016's Postgres work, which has
  no equivalent in-process fake available at the same fidelity.
- **A real bug caught by writing the shared test fixture, not
  assumed safe**: an early version of the `ingest_sync` test helper
  let a Corpus-under-test and the worker's own view of `get_corpus()`
  silently diverge whenever a test used a *different* Corpus per
  client variant (`client_with_generator` vs. the default `client`) —
  caught by running it, not inspection, and fixed by making the test
  helper take the target `Corpus` as an explicit argument rather than
  resolving it through same-named-fixture magic.
- **A real, intentional behavior change**: an unsupported file
  extension (previously a synchronous `400`) now always enqueues
  successfully and fails *as the job's own status* — the upload step
  has no way to know the extension is unsupported before a `Loader`
  actually tries to load it. `test_unsupported_file_extension_fails_the_job_not_the_enqueue`
  pins this down explicitly rather than leaving it to be rediscovered
  as a surprise.
- `POST /ingest`'s response contract is now `202` + `{job_id,
  source_id, status}`, not `200` + the full `IngestResponse` body —
  every existing caller of the old contract needs updating: this
  pass's own test suite (test_api.py/test_auth.py), plus `mcp-server/`
  (unaffected — it calls `Corpus.ingest()` directly, never through
  this HTTP contract), both SDKs, and `run_test_server.py`/the
  TypeScript SDK's own E2E suite, all tracked as explicit, not-yet-
  done follow-on work from this ADR, not silently left broken.
