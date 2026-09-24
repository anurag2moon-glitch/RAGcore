# Integration test suite: findings

A permanent, CI-worthy integration suite now lives under `tests/integration/`
(~2,700 lines) exercising ragcore at production scale — 8 full-length
synthetic books plus, optionally, 8 real Project Gutenberg books, ingested
into one shared `LanceDBStore` + `SqliteGraphStore`, queried under load, and
subjected to concurrent writers and injected failures. Everything runs with
deterministic mock embedders/extractors: zero network calls, zero cost, and
it exercises the *real* `StructuralChunker`, `LanceDBStore`, `SqliteGraphStore`,
`build_graph`, and `GraphRetriever` — only the embedding model and the LLM
extraction call are doubled.

**Baseline before this work:** `pytest -q` → 41 passed (single-book fixture,
7 chapters of *Pride and Prejudice*). **After:** 76 passed by default
(`pytest -q`, ~7.7s), plus 4 more under `pytest -m gutenberg` against real
downloaded books. All 41 original tests pass unchanged.

Four real defects were found and fixed. One more was found, deliberately
**not** fixed, and is flagged below for a decision. One interaction between
two of the fixes was also found and is documented as a caveat.

## What broke

### 1. `SqliteGraphStore.neighbors()` silently dropped its `rel_types` filter on one side

`ragcore/graph/sqlite_store.py` built:

```sql
WHERE source_entity_id IN (...) OR target_entity_id IN (...) AND relation_type IN (...)
```

SQL's default precedence makes this `source IN (...) OR (target IN (...) AND
type IN (...))` — a match on the *source* side was returned regardless of its
`relation_type`. Verified directly: querying `neighbors([a], rel_types=["likes"])`
against a graph with both a `likes` and a `hates` relation from `a` returned
both. Every existing unit test happened to query in a way that never
exercised the source-side branch with a real type mismatch.

**Fix:** parenthesize the OR clause. Regression test:
`tests/integration/test_cross_book_entities.py::test_neighbors_rel_types_filters_both_traversal_directions`.

### 2. `SqliteGraphStore` was thread-hostile, and unsafe under multi-process writers too

Two related problems:

- `sqlite3.connect()` used the default `check_same_thread=True` — any use
  from a second thread raised `ProgrammingError` outright. Verified directly.
- `add_entities`/`add_relations` do a read-then-write merge
  (`SELECT ... ; INSERT/UPDATE ...`) with no transaction guard. Two writers
  merging into the *same* entity/relation id concurrently could both read
  "not present yet" and both insert, or both read stale `extra` and clobber
  each other's aliases — a lost-update race. This matters a lot for
  concurrent multi-book ingestion, which is exactly the scenario where two
  books mention the same cross-book entity "at the same time."

**Fix:** `check_same_thread=False` + `journal_mode=WAL` + `busy_timeout=30000`
+ a `threading.Lock` (serializes same-instance multi-thread callers) + an
explicit `BEGIN IMMEDIATE`/`COMMIT`/`ROLLBACK` around each merge (serializes
*separate* connections to the same file — real for multi-process ingestion).
Regression tests: `tests/integration/test_concurrency.py::test_concurrent_graph_writers_same_store_instance_no_lost_updates`
and `::test_concurrent_graph_writers_separate_connections_same_file` — 40 and
20 concurrent writers merging into one shared entity, asserting every
writer's chunk id and alias survives.

### 3. `GraphRetriever`'s graph expansion was dead code whenever vector search alone could fill `k`

`ragcore/retrieval/graph.py` always scored every expanded (graph-reached)
result at `min_seed_score * decay` — strictly below every seed — then took
`merged[:k]`. The moment vector search returns `k` or more seeds (the normal
case once a corpus is bigger than one query's semantic neighborhood),
`merged[:k]` truncates away *every* expansion. Graph traversal only ever
surfaced when the vector store had fewer than `k` chunks total. This is
exactly "correct at scale, not just the 7-chapter case" — every one of the
three existing `GraphRetriever` unit tests used exactly one fixed seed, so
this was invisible.

**Fix (per your direction — reserve a slice of k for graph hits):**
`GraphRetriever` now fetches `inner.retrieve(query, k=k)` once, uses the
first `k - reserve` as seeds for entity/neighbor lookups, and treats the
remainder as a same-call fallback pool that backfills the result set if graph
expansion contributes less than the reservation (or nothing at all) — so a
query with no graph signal still gets a full `k` vector-only results, and the
"never more than k" contract every caller assumes still holds. Default
reserve is `max(1, k // 3)`, overridable via a new `graph_reserve` param.
Regression test: `tests/integration/test_query_load.py::test_graph_hybrid_reaches_content_pure_vector_search_misses`
— confirmed to fail against the pre-fix code (verified by temporarily
reverting the fix and re-running) and pass after it.

### 4. `build_graph` entity resolution didn't scale past one book

Two compounding problems in `ragcore/graph/pipeline.py` and
`ragcore/graph/resolver.py`:

- `store.entities_by_type(candidate.entity_type)` was called *inside* the
  per-candidate loop — once per candidate, not once per chapter — even
  though nothing is written back to the store until the whole chapter's
  candidates are resolved, so the query result is identical for every
  candidate of the same type within a chapter.
- `entities_by_type`/`entities_for_chunks` were N+1: one extra query per
  entity to fetch its chunk ids. Measured: 41ms at N=2000 entities, paid
  *once per candidate*.
- `EmbeddingEntityResolver.resolve()` re-embedded **every** existing entity
  of that type on **every** candidate call — O(candidates × entities) in
  embedding calls system-wide, not O(candidates + entities).

At ~10 books this compounds into millions of redundant embedding calls and
made requirements 1 and 2 (many books, cross-book resolution) unreachable in
any reasonable time with a real embedder.

**Fix (semantics-preserving, verified identical resolution decisions):**
`entities_by_type` results are now cached per `(chapter, entity_type)` in
`build_graph`'s loop (the loop's writes still all happen after the cached
value would be consulted, so the cached value is provably identical to what
a fresh query would have returned); the N+1 `entity_chunks`/`relation_chunks`
lookups were collapsed into single batched `IN (...)` queries; and
`EmbeddingEntityResolver` now caches an existing entity's embedding vector
keyed by `(entity_id, entity_text)` — keying on the text (not just the id)
means a stale entity (one whose aliases changed via a merge) naturally
misses the cache rather than ever returning a wrong vector, and cache misses
are still batched into one request alongside the candidate, so the "first
time we see this entity" case doesn't regress into one API call per entity.
All 19 existing graph tests pass unchanged after these fixes, confirming no
behavior change beyond call volume.

## What was found and deliberately *not* fixed

### SQLite WAL mode's sidecar files are load-bearing — and losing them surfaces a raw, unwrapped exception

A direct consequence of fix #2 above: enabling `journal_mode=WAL` means a
graph store's recent writes live in `graph.sqlite3-wal` (and `-shm`) until a
checkpoint, not in the main `.sqlite3` file itself. A `SqliteGraphStore`
written to and then dropped without an explicit close (mirroring a killed
process, or simply Python garbage-collecting it) never checkpoints. Copying,
backing up, or restoring *only* the main `.sqlite3` file — a real, easy
operational mistake — silently discards everything not yet checkpointed.

Worse: reopening the orphaned main file doesn't just show stale data, it
raises a raw `sqlite3.OperationalError: disk I/O error` — **not** a
`ragcore.errors.RagcoreError` subclass, violating `ragcore/errors.py`'s own
stated contract ("All library-raised exceptions derive from RagcoreError so
consuming projects can catch broadly"). Reproduced deterministically (3/3
runs). This gap is **broader than `SqliteGraphStore`**: neither it nor
`LanceDBStore` wraps any underlying backend exception (`sqlite3.*`,
lancedb/pyarrow exceptions) in a typed ragcore error anywhere in the storage
layer — this was already true before this session's fixes, just harder to
trigger without WAL mode's sidecar-file dependency making it easy to
demonstrate.

Documented, not fixed, as
`tests/integration/test_failure_injection.py::test_sqlite_graph_wal_sidecar_loss_is_a_real_unwrapped_failure`
— it currently asserts the *raw* `sqlite3.OperationalError`, i.e. it encodes
today's behavior rather than a desired one. **This needs your decision**:
checkpoint on every write or on an explicit `close()`? Wrap backend
exceptions in `StoreError` throughout the storage layer (a real, if
mechanical, cross-cutting change)? Document the operational requirement
(always ship the `-wal`/`-shm` sidecars) and leave it? I'd lean toward
wrapping exceptions being the highest-value, lowest-risk piece of this if
you want anything done now, but didn't want to make that call unilaterally
given it touches exception-handling in two files outside this session's
approved fix list.

## A found interaction between two working, unmodified features

Not a bug — a documented caveat that the synthetic library's test design had
to route around. `ragcore.graph.pipeline.build_graph` attaches a relation
hint (e.g. `"commands Obadiah"`) to an entity's `extra["relations"]`, which
`EmbeddingEntityResolver._entity_text` folds into the text it embeds for
similarity comparison (this is the documented, intentional mitigation for
relationally-described entities — see `CLAUDE.md`'s "Known limitation"
section). Measured directly: attaching a relation hint to an entity **dilutes
the cosine similarity** the resolver computes for that entity's *other*,
unrelated alias-drift merges, if the drifted mention doesn't share the same
relation context. In one concrete case built while writing this suite,
adding `[commands Obadiah]` to "Captain Aurelia Venn"'s embedded text dropped
her similarity to her own later "Venn"/"Aurelia" mentions from ~0.92/0.90
(comfortably above the 0.87 production default threshold) to ~0.83/0.81
(comfortably below it) — a legitimate alias merge silently stopped
happening, purely because of an unrelated relation-hint feature elsewhere in
the same entity's history. The test suite works around this by not
attaching a relation to any entity that also needs to demonstrate alias-drift
merging (see the comment at `tests/integration/synth/library.py`'s `i == 5`
block). Whether this is worth addressing in the real resolver (e.g. weighting
relation-hint tokens differently, or comparing name/alias similarity
separately from relation similarity) is a design question for you — it's a
real, not hypothetical, quality interaction, but it only manifests on an
entity that both accumulates relation hints *and* needs alias resolution
later, which may or may not be common enough in your actual target corpus to
matter.

## What was verified good (locked in with regression tests, not "fixed")

- **`LanceDBStore` under concurrency**: 8 threads × 400 chunks, both via one
  shared `LanceDBStore` instance and via 8 separate connections to the same
  on-disk table — zero errors, exact chunk counts, no duplicate/missing rows,
  in every run.
- **Large `IN (...)` predicates**: 5,000-id upsert/delete and 2,000-id `get()`
  round-trip correctly and fast (no batching needed at this scale).
- **`get_by_source` on an empty table**: returns `[]` cleanly rather than
  tripping over a `limit(0)` edge case.
- **Malformed-input handling**: every format loader (`epub`, `pdf`, `docx`,
  unsupported extension, directory-as-path) already raised a typed
  `LoaderError` for genuinely broken input; zero-byte/empty files and invalid
  UTF-8 already degraded gracefully (0 chunks / best-effort decode) rather
  than crashing. No changes needed here — the loader layer's existing error
  handling held up under adversarial input.
- **Recovery after a failed ingest**: a corpus that just failed to ingest a
  malformed file is still fully usable for the next, healthy ingest — no
  poisoned state.

## The eval harness (new)

There was **no** `Recall@k`/`nDCG@10`/`MRR` code anywhere in ragcore, the
existing tests, or the 935-line design doc before this session — "re-run the
existing eval harness at larger scale" first meant building one
(`tests/integration/eval/`: `metrics.py`, `queryset.py`, `report.py`), test-only,
not shipped as part of the `ragcore` package (metrics are a testing concern,
not a library API surface).

Ground truth is needle-based: unique marker tokens planted in known chunks
across the synthetic library, so the "relevant chunk" set is computed by
scanning real ingested chunk text, not assumed from authoring-time offsets.
**Scope note**: this measures the *retrieval pipeline* (chunking, storage,
filtering, top-k ordering, hybrid graph expansion) at scale, not embedding
quality — `DeterministicEmbedder` is a lexical hashed bag-of-words double,
not a semantic model, standing in for `voyage-context-4` for cost reasons.

Run at both scales with identical metric code: a 3-book subset and the full
8-book library (the samples/library.lancedb 7-chapter fixture is gitignored
and not guaranteed present, so it wasn't used as the "small scale" anchor —
see `tests/integration/test_eval_quality.py`'s module docstring). Committed
baseline: `tests/integration/baselines/eval_baseline.json` — recall@10=1.0,
nDCG@10≈0.908, MRR=0.875 at full library scale, vector-only and graph-hybrid
identical (as expected: the graph-hybrid scenario is a separate, deliberately
constructed probe, not one of the needle queries).

## Suite structure

```
tests/integration/
  synth/generator.py   deterministic themed-prose generator (6 themes, seeded)
  synth/library.py     8 book archetypes + planted needles/entities/graph probe
  doubles.py           DeterministicEmbedder, MockEntityExtractor, fault injectors
  eval/                metrics.py, queryset.py, report.py
  conftest.py          session-scoped shared_library fixture; RAGCORE_IT_SCALE knob
  test_ingest_library.py        req 1 — many books, one shared index
  test_cross_book_entities.py   req 2 — cross-book resolution + merge/separate
  test_concurrency.py           req 3 — concurrent ingestion, no corruption
  test_query_load.py            req 4 — load, latency, hybrid retrieval at scale
  test_failure_injection.py     req 5 — kill mid-file, corrupt, ENOSPC, malformed input
  test_eval_quality.py          req 6 — Recall@k/nDCG/MRR at both scales
  test_gutenberg.py             opt-in: real books (pytest -m gutenberg)
```

Run: `pytest -q` (includes the full integration suite by default, ~7.7s,
excludes `live` and `gutenberg`). `RAGCORE_IT_SCALE=quick` for fast local
iteration (4-book subset). `python scripts/fetch_gutenberg.py` then
`pytest -m gutenberg` for the real-book layer (verified working against 8
real downloads in this session).
