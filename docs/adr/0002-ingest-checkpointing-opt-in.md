# 2. Ingest-stage checkpointing is opt-in, not the default

Date: 2026-10-01

## Status

Accepted

## Context

`docs/ARCHITECTURE.md`'s gap list (Phase 1, item 3) flagged that
`Corpus.ingest()` has no stage-level resumability: a large file failing
partway through embedding re-runs load -> chunk -> embed from scratch on
retry. Every other LLM-backed pass in this codebase (concepts, contextual,
hierarchy, claims) already solves this with the same shape — a
fingerprinted checkpoint per unit of work (one chapter/`ChunkGroup`),
resumed by default — but as a separate, opt-in construction step the
caller explicitly wires in after `ingest()`, never inside it.

`Corpus.ingest()` itself is different: it is the one pipeline every caller
already depends on with no setup at all, and today it embeds every
`ChunkGroup` and upserts the whole resulting chunk list in one call at the
end. Making a group resumable requires upserting each group into the
VectorStore as soon as it's embedded, instead of batching every chunk into
one final `store.upsert(chunks)` call — a real, observable change: a
reader calling `corpus.retrieve()` while a large multi-chapter ingest is
still running would, under the new scheme, see earlier chapters already
present before the file finishes ingesting.

## Decision

Mid-ingest checkpointing is an additive, optional capability:
`Corpus(..., checkpoint_store: IngestCheckpointStore | None = None)`
(`ragcore.ingestion.SqliteIngestCheckpointStore` by default-available
implementation). When `checkpoint_store` is `None` (the default),
`Corpus.ingest()` behaves exactly as before — single-shot, one final
upsert, no new SQLite file created, no behavior change for any existing
caller or test. When a `checkpoint_store` is supplied, `ingest()` embeds
and upserts one `ChunkGroup` at a time, checkpointing each immediately
after a successful upsert, and a retried `ingest()` call on the same file
skips every group whose checkpoint fingerprint still matches.

The fingerprint is **not** `sorted(chunk_ids)` alone, unlike the
concepts/contextual precedent. `Chunk.chunk_id` here is positional
(`f"{source_id}:{chunk_index}"`, see `ragcore/chunking/structural.py`),
not content-addressed — two different chunker configurations over the
same file can produce the identical chunk_id sequence with different
actual text at each index. The fingerprint therefore hashes the actual
chunk texts (plus the embedder's `name` and any `metadata` passed to
`ingest()`), so a chunker, embedder, or metadata change correctly
invalidates every checkpoint it affects rather than silently resuming
onto stale content. See `ragcore/ingestion/pipeline.py` for the exact
construction.

`ingest_many()` gains `max_retries: int = 0` (default: today's
single-attempt behavior). With a `checkpoint_store` set and
`max_retries > 0`, a failing file is retried in place — each retry resumes
from its own last-completed group rather than restarting — and after
exhausting retries is recorded via `checkpoint_store.mark_dead_letter()`
in addition to the existing best-effort `errors` list, so an operator has
a persistent, inspectable list of permanently-failing files rather than
only an in-memory report from one `ingest_many()` call.

## Consequences

- Zero behavior change, zero new dependency, for the large majority of
  existing and future callers that don't pass `checkpoint_store` — this
  satisfies "extend, don't rewrite working code."
- Callers who do want resumability must explicitly provide a
  `checkpoint_store` and accept the early-partial-visibility tradeoff
  described above (chunks become queryable as each group lands, not only
  once the whole file finishes). This is the right tradeoff for the kind
  of file this matters for (a large book, embedded chapter-by-chapter,
  where partial visibility of already-finished earlier chapters is
  harmless and arguably desirable) and is documented on the constructor
  argument and in `ragcore/ingestion/__init__.py`.
- `tenant_id`-scoped dead-letter handling (one dead-letter list per
  tenant, not one per `IngestCheckpointStore` instance) is deferred to
  Phase 7, same as every other tenant-scoping decision in
  `docs/ARCHITECTURE.md`'s gap list — this ADR's `IngestCheckpointStore` is
  single-tenant-shaped today, matching the rest of the codebase.
