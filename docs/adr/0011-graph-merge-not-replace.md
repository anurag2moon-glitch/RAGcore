# 11. GraphStore/ConceptStore writes merge; VectorStore writes replace

Date: 2026-10-07 (backfilled — see "Status" below)

## Status

Accepted. **Backfilled**: this decision predates the ADR practice in
this repository; recorded now per `docs/ARCHITECTURE.md`'s Phase 9 ADR
backfill gap item, since it is the one place this codebase's two
storage-layer write semantics diverge, and the divergence is easy to
get backwards if a new `GraphStore`/`ConceptStore` implementation is
written by analogy to `VectorStore.upsert()` without re-deriving why.

## Context

`VectorStore.upsert()` is replace-by-id: writing a chunk with an
already-existing `chunk_id` overwrites the old row outright, because a
chunk's content genuinely *is* replaced when a source is re-ingested
(see `docs/adr/0010-content-addressed-identity.md` for `source_id`'s
content-addressing — a changed file gets a wholly new `source_id`, so
"the same chunk_id with different content" shouldn't even arise for a
single source in practice, and when it's deliberately re-upserted,
e.g. after contextual-embedding re-vectorization, replacement is
exactly the intended behavior).

An entity or concept is different in kind, not just in schema: "Bingley"
mentioned in chapter 1 and "Mr. Bingley" mentioned in chapter 3 of the
same book resolve to the *same* `entity_id` (content-addressed over
normalized name — ADR 0010) precisely because they're the same
real-world person, restated. If `add_entities()` behaved like
`upsert()`, the second write would silently delete the first chapter's
provenance (`source_chunk_ids`) the moment the same entity recurs —
the exact opposite of what GraphRAG extraction needs, since the whole
value of the graph is accumulating *every* chapter that mentions an
entity, not just the most recent one.

## Decision

`SqliteGraphStore.add_entities()`/`add_relations()` and
`SqliteConceptStore`'s equivalent writes **merge** into an existing row
when the (content-addressed) id already exists, rather than replacing
it: `source_chunk_ids` and `extra["aliases"]` are unioned, never
overwritten (`ragcore.graph.merge.merge_extra`, shared between the
SQLite and Kuzu backends specifically so this isn't an accident of one
implementation). `VectorStore.upsert()` keeps replace-by-id semantics
unchanged. Any future `GraphStore`/`ConceptStore` implementation must
match the merge contract — `SqliteGraphStore` is the reference
implementation to diff a new one against, not `LanceDBStore`.

This has a real, tested edge: content-addressed ids make the *common*
case of exact-name recurrence free (no resolver call at all — an exact
canonical-name match just merges), but it does **not** mean cross-chapter
aliasing is rare in practice. Tested against real chapters of *Pride
and Prejudice*: "Bingley" (ch. I) and "Mr. Bingley" (ch. II/III) are the
same person, independently canonicalized per chapter — needing
`EmbeddingEntityResolver` to merge them, and needing it *again* in
chapter III even though chapter II had already resolved it, since a
fresh extraction call regenerates a candidate's own id from its own
chosen name every time, regardless of what it previously got merged
into. This resolver is a frequently-exercised path, not a rare
fallback (`CLAUDE.md`'s own GraphRAG section records this finding in
full, including the known, documented limitation for entities
introduced only by relational description rather than by name — "the
eldest Miss Bennet" never getting linked to "Jane" without a shared
relation to overlap on).

## Consequences

- A chapter's extraction can be safely re-run (after a model change, a
  prompt-version bump, or simply a retry) without manually diffing
  against what's already stored — the merge contract makes that safe
  by construction, the same way content-addressed ids make a
  `build_*()` pass idempotent (ADR 0010).
- A new `GraphStore` or `ConceptStore` backend (the way `KuzuGraphStore`
  was added as a second implementation) must independently implement
  this exact merge contract; nothing in the `GraphStore`/`ConceptStore`
  Protocol *type* enforces it mechanically today — only the reference
  implementation's own behavior and test suite do, which is
  consequential enough to call out here rather than leave as an
  implicit expectation a new implementation could miss.
- The cross-chapter resolution cost (an `EmbeddingEntityResolver` call
  on most non-trivial extractions, not just a rare one) is an accepted,
  measured cost of this design, not a bug to optimize away by trying
  to make exact-name matching cover more cases than it safely can.
