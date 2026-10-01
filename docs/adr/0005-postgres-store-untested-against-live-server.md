# 5. PostgresIngestCheckpointStore is untested against a live Postgres in this environment

Date: 2026-10-01

## Status

Accepted

## Context

`docs/ARCHITECTURE.md` gap list Phase 2 item 12 asks for "Postgres for
metadata, jobs" alongside S3-compatible raw file storage. Every other
default store implementation in this codebase (`SqliteGraphStore`,
`LanceDBStore`, ...) is tested against its real backend unconditionally,
because the real backend is either embedded (SQLite, stdlib) or runs
fully in-process with no external service (LanceDB; S3 via `moto`'s
in-process mock, added alongside this ADR). Postgres has no equivalent:
it is a real server process, and this development environment has no
Docker, no package manager (`brew`/`apt`/`conda` all absent), and no
Postgres binary already installed — there was no way to stand one up to
test against in this pass.

Shipping a Postgres store with zero real-backend verification would be
inconsistent with every other store in this codebase, all of which are
exercised against their real backend before being considered done. Not
shipping it at all would leave the "Postgres metadata/job store" gap
item entirely unaddressed, and "Postgres for metadata, jobs" is
explicitly named in the original spec as prerequisite, sequenced work
("before Phase 7 proper").

## Decision

Implement `PostgresIngestCheckpointStore`
(`ragcore/ingestion/postgres_store.py`) as a full, real implementation of
the already-existing `IngestCheckpointStore` protocol — not a stub —
using `psycopg` 3, with schema and method contracts mirroring
`SqliteIngestCheckpointStore` one-for-one. Scoped to this one protocol
rather than writing a Postgres adapter for every SQLite-backed store in
the codebase (graph, concepts, claims, contextual, hierarchy): proving
the pattern once, against the store most directly named by the gap item
("jobs" ≈ ingest checkpointing/dead-letters), is the right size for this
pass — the same "second provider proves swappability" reasoning as
`docs/adr/0003`, not an exhaustive migration.

The test suite (`tests/test_postgres_ingest_checkpoint_store.py`) mirrors
`tests/test_ingest_checkpoint_store.py`'s cases one-for-one, gated behind
a new `postgres` pytest marker (excluded by default, same convention as
the existing `live`/`gutenberg` markers) and `RAGCORE_TEST_POSTGRES_URL`
— it skips cleanly when unset, exactly like `gutenberg`'s local-cache
check already does for its own external dependency.

Before writing the SQL, I sanity-checked the implementation's Python-level
correctness against psycopg 3's real API shape using a mocked connection
(every method's `cursor()`/`transaction()`/`execute()` call pattern
executes without a `TypeError`/`AttributeError`) — this verifies the
Python code is well-formed and calls psycopg correctly, but **does not
verify the SQL itself executes correctly against a real Postgres server**,
which requires an actual server this environment doesn't have.

## Consequences

- This file carries materially lower confidence than every other store in
  this codebase: it has been written carefully (the SQL reuses the exact
  `ON CONFLICT ... DO UPDATE SET` upsert pattern already proven in every
  SQLite store, which is standard, shared Postgres/SQLite syntax) and
  sanity-checked at the Python-API level, but has not been run against a
  real Postgres server by anyone in this session. Treat it as
  implemented-but-unverified-by-execution until it is run once against a
  live server (`RAGCORE_TEST_POSTGRES_URL=postgresql://... pytest -m postgres`).
- `docs/ARCHITECTURE.md` states this caveat plainly rather than reporting
  Phase 2 item 12 as fully, equivalently done — it is real, complete code
  (not a placeholder/stub, satisfying the "no placeholder code" rule),
  but its test-confidence level is explicitly different from the rest of
  this codebase's stores, and that difference should stay visible rather
  than being smoothed over.
- Only one protocol (`IngestCheckpointStore`) has a Postgres
  implementation. Extending the same pattern to `GraphStore`,
  `ConceptStore`, etc. remains open work, to be done (and tested against
  a real server) when it's actually needed — likely alongside Phase 7/8's
  multi-tenant, multi-node deployment target, where Postgres's real
  advantage over SQLite (concurrent multi-process/multi-host access)
  actually matters.
