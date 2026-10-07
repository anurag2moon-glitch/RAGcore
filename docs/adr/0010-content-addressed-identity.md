# 10. Content-addressed ids, with one known gap in chunk_id

Date: 2026-10-07 (backfilled — see "Status" below)

## Status

Accepted, with a known, documented, deliberately-unfixed gap. **Backfilled**:
this scheme predates the ADR practice in this repository; recorded now
per `docs/ARCHITECTURE.md`'s Phase 9 "ADR backfill" gap item, since it
is the single mechanism every later layer's idempotency, resumability,
and cross-layer joining depends on.

## Context

Nearly every store in this codebase (vector, graph, concepts, claims,
lexical, contextual, hierarchy) needs the same property: re-running the
same work over unchanged input must be a no-op, not a duplicate. A
random or sequential id can't give you that — two runs over the same
file would mint two different ids for "the same thing," and every
downstream join (which entity came from which chunk, which concept
links to which claim) would have to go through an external mapping
table instead of the id itself.

## Decision

Identity is derived from content, not generated, everywhere it's
possible to do so cleanly:

- `Document.source_id` / `ChunkMeta.source_id` — `sha256` of the file's
  own bytes. Re-ingesting byte-identical content reproduces the same
  id; a changed file gets a brand-new one. (This is also why two
  tenants ingesting byte-identical files collide on `source_id` — a
  known, accepted edge case named explicitly in
  `docs/adr/0006-tenant-id-foundation-not-enforcement.md` and reused by
  `ragcore.tenancy.TenantScopedCorpus`'s own docstring rather than
  re-litigated there.)
- `entity_id` (`ragcore/graph/`) — `f"entity:{type}:{normalize(name)}"`.
  Deliberately content-addressed over *semantic* text, not scoped to a
  source, specifically so the same real-world entity mentioned in
  chapter 1 and chapter 12 — or in two different books — collapses to
  one row with accumulated provenance (`CLAUDE.md`'s GraphRAG section;
  the merge-not-replace write contract this enables is `SqliteGraphStore`'s
  own, separately documented, discipline). The same shape applies to
  `concept_id` (`ragcore/concepts/`) and community ids.
- `claim_id` (`ragcore/claims/`) — `sha256(source_id | chunk_id |
  normalized_text)`. Scoped to the originating chunk, unlike
  entity/concept ids — two different sources independently asserting
  the same claim must stay two separate rows, since the whole point of
  this layer is capturing *agreement between independent sources*, and
  hashing text alone would destroy that signal before it could ever be
  observed.
- `node_id` (`ragcore/hierarchy/`) — positional
  (`hierarchy:chapter:{source_id}:{chapter_index}`), not content-addressed
  at all. A hierarchy node's identity is inherently tied to one exact
  position in one book's table of contents, unlike a concept, which can
  legitimately recur across completely different sources.

**The one deliberate exception, with a known gap**: `Chunk.chunk_id` is
`f"{source_id}:{chunk_index}"` — positional *within* a content-addressed
source, not a hash of the chunk's own text. Its own docstring overstates
this (says "stable and content-addressed" without the caveat) — see
`docs/ARCHITECTURE.md`'s fragility list, item 10, for the precise gap:
two different chunker configurations over the same file could produce
an identical chunk_id sequence with different text at each index.
Harmless in practice today (nothing relies on chunk_id detecting a
text change at a fixed position), and the one place it would have
mattered — `ragcore.ingestion.pipeline.group_fingerprint`'s checkpoint
fingerprint — was built to hash chunk *text*, not `chunk_id`, from the
start, specifically to route around this gap (see that module's own
docstring and `docs/adr/0002-ingest-checkpointing-opt-in.md`).

## Consequences

- Re-running any `build_*()` pass (`build_graph`, `build_concepts`,
  `build_claims`, `build_hierarchy`, `build_contextual_embeddings`)
  over an unchanged corpus is idempotent for free — no separate
  dedup pass, no external mapping table. This is the single property
  every checkpoint/resume mechanism in this codebase is built on top
  of, not a nice side effect.
- Every cross-layer join (which chunks support this entity, which
  concept does this claim anchor to) is a plain id comparison, with no
  schema needed beyond "store the id."
- The `chunk_id` docstring's overstated guarantee (item 10 above)
  remains an open, tracked correction — worth fixing the next time
  that file is touched for an unrelated reason, deliberately not
  patched here just to close an ADR.
