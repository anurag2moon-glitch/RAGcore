# Operator runbook

Phase 9 deliverable. Currently one section — backup/restore — the
Phase 8 gap list item this was written to close
(`docs/ARCHITECTURE.md`'s Phase 8 item 38: "the mechanism already
exists per-store; only the written procedure is missing"). Add further
sections here as other operational procedures need writing up.

## Backup and restore

Every fact below was verified directly against the code and its test
suite before being written here — exact method signatures, exact file
paths, and the restore behavior LanceDB's own tag/version mechanism
actually has (confirmed via `tests/test_lancedb_snapshot.py`, not
assumed from a docstring). Where the library does **not** automate
something, that's called out explicitly in its own section below
rather than left implicit.

### What `ragcore.versioning.snapshot` actually does

`create_snapshot()` and `rollback()` (`ragcore/versioning/snapshot.py`)
orchestrate a snapshot across up to seven stores in one call: the
vector store plus six optional SQLite-backed stores — graph, concepts,
hierarchy, context, lexical, claims. Each store argument is duck-typed
(`hasattr(store, "backup_to")`/`hasattr(store, "snapshot")`) — pass
only the stores your deployment actually uses; an omitted store is
silently skipped, not an error.

```python
from ragcore.versioning.snapshot import create_snapshot, rollback

snap = create_snapshot(
    tag,                              # a tag string you choose, e.g. a timestamp
    manifest,                         # IngestionManifest instance
    vector_store,                     # LanceDBStore
    snapshot_dir="/path/to/snapshots",
    graph_store=graph_store,          # SqliteGraphStore, if you have one wired
    concept_store=concept_store,      # SqliteConceptStore
    hierarchy_store=hierarchy_store,  # SqliteHierarchyStore
    context_store=context_store,      # SqliteContextStore
    lexical=lexical_store,            # Fts5LexicalStore
    claim_store=claim_store,          # SqliteClaimStore
    max_snapshots=10,                 # retention: oldest LanceDB tags beyond this are dropped
)
```

Step by step, this:
1. Writes a `status='pending'` row to the manifest's own `snapshots`
   table *before* touching any store — so a crash mid-snapshot leaves a
   detectable, refuse-to-restore record rather than nothing.
2. Tags the vector store's current LanceDB version (`vector_store.snapshot(tag)`)
   and applies retention (drops the oldest LanceDB tags beyond
   `max_snapshots`, oldest-version-first).
3. Calls `backup_to(snapshot_dir / f"{tag}.{name}.sqlite3")` on every
   SQLite-backed store actually passed in — the stdlib `sqlite3`
   **online backup API**, safe under concurrent readers in WAL mode
   (never a plain file copy, which can capture a torn snapshot of a
   live WAL-mode file).
4. Flips the manifest row to `status='complete'` and records the real
   LanceDB version plus every backup file's path — **only after every
   backup above has already succeeded.**

```python
rollback(
    manifest, tag, vector_store,
    graph_store=graph_store, concept_store=concept_store,
    hierarchy_store=hierarchy_store, context_store=context_store,
    lexical=lexical_store, claim_store=claim_store,
)
```

This restores the LanceDB table to the tagged version
(`vector_store.restore_snapshot(tag)`) and restores every SQLite store
in place from its backup file (`store.restore_from_backup(path)`,
the same online backup API run in reverse against the live, already-open
connection — no close/reopen needed). It **refuses** if the snapshot's
own status isn't `'complete'`, with a clear error naming the actual
status — exactly the pending-snapshot case step 1 above exists to make
detectable.

A full, runnable worked example (ingest two books, edit one, sync —
which auto-snapshots before the risky step — then roll back and
confirm both the vector store and the graph store reverted) is
`scripts/sync_demo.py`. It's the fastest way to see this end to end
before trusting it against a real deployment; run it with
`python scripts/sync_demo.py`.

### LanceDB's own snapshot semantics (what "restore" actually does)

`LanceDBStore.snapshot(tag)`/`restore_snapshot(tag)`/`list_snapshots()`/
`drop_snapshot(tag)` are a thin wrapper over LanceDB's **native**
tag/version API — confirmed, not assumed, by directly asserting on the
version numbers in `tests/test_lancedb_snapshot.py`:

- **A restore is a new version, not a history rewrite.** Restoring to
  tag T produces a new current version equal to T's own version — the
  version history between "now" and T is never deleted.
- **This makes a rollback itself reversible.** You can tag `before-X`,
  mutate, tag `after-X`, restore to `before-X`, and `after-X` is *still*
  a valid, restorable tag afterward — restoring never invalidates a
  later tag you already took.
- **A tagged version is protected from cleanup.** `compact()`/
  `optimize(cleanup_older_than=...)` **raises** rather than silently
  proceeding if the cleanup window would remove a tagged version. You
  must `drop_snapshot(tag)` first if you actually want that version's
  space reclaimed — `drop_snapshot` removes only the tag, never the
  row data itself (the data becomes reclaimable by a later compaction,
  it isn't deleted by `drop_snapshot`).

### What this does **not** automate — read before relying on this for a real deployment

1. **The `IngestionManifest` itself is not covered.** `create_snapshot`/
   `rollback` snapshot and restore the *content* stores; they never
   touch the manifest's own SQLite file (only read/write individual
   rows describing *other* snapshots in it). `IngestionManifest` does
   have its own `backup_to(path)` method, but **no matching
   `restore_from_backup`** — and `create_snapshot` never calls even the
   `backup_to` it has. Rolling back your content stores to tag T while
   the manifest's `sources` ledger (used by `plan_sync`/`sync_corpus`'s
   diffing) has moved past T leaves the manifest pointing at
   `source_id`s a just-completed rollback made disappear — a real
   consistency gap. If point-in-time consistency matters for your
   deployment, back up the manifest's SQLite file yourself at the same
   moment as `create_snapshot()` (the same `sqlite3.Connection.backup()`
   pattern every other store already uses, run against the manifest's
   own connection) and restore it in lockstep with `rollback()` — there
   is no library method for this today.
2. **Ingest checkpoint stores are excluded.** Neither
   `SqliteIngestCheckpointStore` nor `PostgresIngestCheckpointStore` is
   one of the six stores `create_snapshot`/`rollback` know about.
   `SqliteIngestCheckpointStore` does have its own `backup_to`/
   `restore_from_backup` — call them yourself and coordinate the tag
   and timing manually if you need this checkpoint state to roll back
   too. `PostgresIngestCheckpointStore` has **no backup facility at
   all** — its own module docstring states this is deliberately left
   to standard Postgres tooling (`pg_dump`, WAL archiving); see
   `docs/adr/0005-postgres-store-untested-against-live-server.md` for
   the related caveat on that store more generally.
3. **The LanceDB tag and each SQLite backup are not one atomic
   cross-store transaction.** They run sequentially; the pending/complete
   flag exists specifically to make a crash mid-sequence detectable and
   non-restorable, not to make the sequence itself atomic. Nothing
   pauses concurrent ingestion/writers while a snapshot is in progress
   — avoid calling `create_snapshot()` concurrently with an active
   `ingest()`/`sync_corpus()` against the same stores if you need a
   genuinely point-in-time-consistent snapshot.
4. **You must pass the same set of stores to `rollback()` that you
   passed to `create_snapshot()`.** Every store argument is duck-typed
   with a silent skip if omitted — there is no validation that the
   stores you're rolling back with match what you snapshotted. Omitting
   `claim_store` from a `rollback()` call that had it at snapshot time
   silently leaves the claim store un-rolled-back, with no error or
   warning.
5. **Retention (`max_snapshots`) only prunes LanceDB tags.** It does
   not delete old `.sqlite3` backup files under `snapshot_dir`, nor old
   rows in the manifest's own `snapshots` table. Disk space management
   for those is an operator task this library does not handle.
