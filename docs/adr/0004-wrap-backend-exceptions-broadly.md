# 4. Wrap storage-layer backend exceptions broadly, not per-backend-type

Date: 2026-10-01

## Status

Accepted

## Context

`docs/ARCHITECTURE.md` §3 (fragility item 1, carried over from
`docs/INTEGRATION_TEST_FINDINGS.md`) documents a real, reproduced gap:
`errors.py` states "All library-raised exceptions derive from
RagcoreError," but no storage-layer implementation
(`SqliteGraphStore`, `SqliteConceptStore`, `SqliteClaimStore`,
`SqliteContextStore`, `SqliteHierarchyStore`, `Fts5LexicalStore`,
`IngestionManifest`, `LanceDBStore`) actually wraps its backend's
exceptions — a lost SQLite WAL sidecar file raises a raw
`sqlite3.OperationalError`, not a `StoreError`, and nothing stops the same
gap for LanceDB/pyarrow exceptions. A prior session found and deliberately
deferred this, flagging "wrap backend exceptions in StoreError throughout
the storage layer" as the highest-value, lowest-risk fix available,
without picking an exact mechanism.

Two ways to close it: (1) enumerate each backend's specific exception
types (`sqlite3.Error` and subclasses; whatever LanceDB/pyarrow raises)
and catch only those, narrowly; or (2) catch broadly (`except Exception`,
explicitly re-raising anything already a `RagcoreError` unchanged) at
every public store method.

Narrow, per-backend-type catching requires confidently enumerating every
exception type two different native libraries (sqlite3's C extension,
lancedb/pyarrow's Rust-backed bindings) can raise — missing one silently
reintroduces exactly the gap being closed, and that enumeration isn't
something this pass can verify exhaustively against both libraries'
actual behavior. Broad catching has an opposite, more contained risk: a
genuine ragcore programming bug (e.g. a caller type error) inside a
decorated method would also get wrapped into a `StoreError` instead of
surfacing as its original exception type — but the library's own stated
contract already promises `RagcoreError` broadly, and a wrapped-but-still-
informative `StoreError(f"{fn.__qualname__} failed: {e}")` (original
exception chained via `from e`, never swallowed) is strictly more useful
to a caller than an exception type the library's own docs say can't
happen.

## Decision

One decorator, `ragcore.errors.wrap_backend_errors`, applied to every
public method on every storage-layer class listed above (plus
`ragcore.ingestion.SqliteIngestCheckpointStore` and
`ragcore.embedding.cache.SqliteEmbeddingCache`, added after this gap was
first documented but with the identical gap). It catches `Exception`,
re-raises any `RagcoreError` unchanged (so an already-intentional typed
raise inside a method — e.g. `LanceDBStore`'s filter-allowlist
`StoreError` — is never double-wrapped or its message altered), and wraps
everything else as `StoreError(f"{fn.__qualname__} failed: {e}") from e`
(`from e` preserves the original traceback and exception for anyone who
inspects `.__cause__`). Private (`_`-prefixed) helper methods are left
undecorated — they're only ever called from an already-decorated public
method in the same class, so wrapping them too would be redundant, not
incorrect.

## Consequences

- Closes the fragility gap uniformly, in one pass, with no per-backend
  exception-type enumeration to get wrong or leave incomplete.
- A consuming project's `except RagcoreError:` now actually catches
  everything this layer can raise, matching `errors.py`'s existing
  documented promise for the first time.
- The one accepted tradeoff: a real ragcore bug surfacing as a
  `StoreError` instead of its original exception type is slightly less
  immediately diagnosable from the exception's class alone — mitigated by
  always chaining the original exception (`from e`) and including
  `fn.__qualname__` and the original message in the `StoreError` text, so
  the original failure is still one `.__cause__` away, not lost.
- `tests/integration/test_failure_injection.py::test_sqlite_graph_wal_sidecar_loss_is_a_real_unwrapped_failure`
  encoded the *old*, undesired behavior (asserting a raw
  `sqlite3.OperationalError`) — updated in this pass to assert the now-
  correct `StoreError`, per that test's own docstring inviting exactly
  this once the gap was fixed.
