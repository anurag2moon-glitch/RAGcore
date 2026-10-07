# 7. tenant_id on the remaining stores splits into two genuinely different problems

Date: 2026-10-07

## Status

Accepted

## Context

Phase 7 gap item 28 ("`tenant_id` on every other store") reads, at a
glance, like six repetitions of the same bolt-on already done for
`LanceDBStore` in commit 7b7393d — add a nullable column, add it to a
filterable-fields allowlist, migrate an existing on-disk table. Checking
each store's actual id scheme before starting shows that isn't true: the
six remaining stores split into two tiers with a real correctness
difference between them, not just a difference in how much code each one
has.

**Tier 1 — id is already scoped by `chunk_id`/`source_id`, so a passive
column is correct.** Verified directly in each store's own code, not
assumed:
- `Fts5LexicalStore` (`ragcore/lexical/fts5.py`) — FTS5 virtual table
  rows keyed by `chunk_id`.
- `ragcore/contextual/sqlite_store.py`'s context store — `context_blurbs`
  table, `chunk_id TEXT PRIMARY KEY`.
- `ragcore/hierarchy/` — node ids are positional
  (`hierarchy:chapter:{source_id}:{chapter_index}`, etc., per
  `ragcore/hierarchy/pipeline.py`'s own module docstring), not
  content-addressed across sources.
- `SqliteClaimStore` (`ragcore/claims/`) — `claim_id =
  _claim_id(source_id, chunk_id, text)` (`ragcore/claims/claude.py`).

For all four, adding a nullable `tenant_id` column plus a filter, exactly
mirroring `LanceDBStore`'s migration (`add_columns`-equivalent `ALTER
TABLE ... ADD COLUMN`, backfilling existing rows with `NULL`), is
sufficient and correct. The only residual risk is the one already
documented for chunks themselves and for `TenantScopedCorpus.ingest()`
(`docs/adr/0006`): two tenants ingesting byte-identical source files
collide on the shared content-addressed `source_id`/`chunk_id` — a
pre-existing characteristic of content-addressing, not something this
tier's `tenant_id` column introduces or needs to solve.

**Tier 2 — id is content-addressed over bare semantic text, with *no*
source scoping at all, by design.** Verified directly, not assumed:
- `ragcore/graph/` — `entity_id = f"entity:{type}:{normalize(name)}"`,
  `relation_id` similarly built from the two endpoint entity ids plus
  type (`CLAUDE.md`'s GraphRAG section). This is deliberate: the whole
  point is that "Bingley" mentioned in chapter 1 and "Mr. Bingley" in
  chapter 3 — or the same real-world entity across two different books —
  collapse to one row, accumulating `source_chunk_ids`/aliases as
  provenance (`SqliteGraphStore`'s "merge, not replace" contract).
- `ragcore/concepts/` — `concept_id = f"concept:{_normalize(label)}"`
  (`ragcore/concepts/claude_extractor.py::_concept_id`). Same shape,
  same reason: a concept like "consequences of unchecked ambition"
  recurring across chapters or books is supposed to merge into one node.

Adding a passive `tenant_id` column to either store's `entities`/
`concepts` table would **not** isolate tenants: two tenants each
producing an entity named "John Smith" (or a concept both happen to
phrase identically after normalization) still compute the *same*
content-addressed id, so `add_entities`/extraction's merge-not-replace
write path still merges them into one shared row regardless of what a
`tenant_id` column says — whichever tenant wrote second would silently
accumulate the first tenant's `source_chunk_ids` and aliases into a row
a `tenant_id` column can only hold one value for anyway. Reading one
tenant's graph would surface the other tenant's provenance chunk ids,
and `GraphRetriever`'s neighbor expansion would walk straight across the
tenant boundary. This is a materially different, worse failure mode than
"forgot to filter" — it is active cross-tenant merging at write time, not
a passive leak a filter could have caught.

## Decision

Treat the two tiers differently rather than force one mechanism onto
both:

- **Tier 1 stores**: implement the `LanceDBStore`-style passive
  `tenant_id` column + filter + migration, store by store, as originally
  scoped for gap item 28. No open design question here — do this next.
- **Tier 2 stores (graph, concepts)**: do **not** bolt on a passive
  column. Isolating these correctly requires baking `tenant_id` into the
  content-addressed id itself (e.g. `entity:{tenant_id}:{type}:{name}`),
  which is a real design decision with its own tradeoffs this ADR
  deliberately does not resolve yet:
  - A `tenant_id`-prefixed id changes every existing entity/relation/
    concept id's format — a breaking change for any consumer that
    persisted or compared against today's ids, and for `None` (no
    tenant) it needs a defined, stable placeholder segment rather than
    an empty one that silently shifts the id's shape.
  - Alternative: keep the id as-is but make the primary key a compound
    `(entity_id, tenant_id)`, with every read/write path's `WHERE
    entity_id = ?` becoming `WHERE entity_id = ? AND tenant_id = ?`.
    SQL's `NULL != NULL` means a compound `UNIQUE`/`PRIMARY KEY`
    including a nullable `tenant_id` column does **not** actually
    enforce "at most one row per entity_id for the single-tenant (NULL)
    case" the way a plain `PRIMARY KEY (entity_id)` does today — that
    needs its own resolution (e.g. a sentinel non-NULL default instead
    of `NULL`), not a mechanical column addition.
  - Either option touches `add_entities`/`add_relations`'s merge logic,
    `neighbors()`/`entities_by_type()`/every read path, and the resolver
    (`EmbeddingEntityResolver`) that today already treats
    content-addressed exact-match as a free, zero-call case — multi-
    tenant correctness must not change single-tenant behavior or cost.

This is intentionally left as an open design question for a dedicated
follow-on pass, not attempted as part of this ADR, rather than ship a
passive column that would look done in a status report while leaving
the actual cross-tenant merge-at-write-time hole wide open.

## Consequences

- Gap item 28 in `docs/ARCHITECTURE.md` is split into two sub-items so
  the tracked status reflects two genuinely different amounts of
  remaining risk, not one undifferentiated "not done yet."
- `TenantScopedCorpus` (`docs/adr/0006`, commit 55f5410)'s own
  documented scope limit — "`GraphRetriever`/`ConceptRetriever` read
  their own stores, none of which carry `tenant_id` yet" — now has a
  precise reason attached: it is not simply unimplemented, the
  straightforward implementation would be actively unsafe for exactly
  the two stores those retrievers depend on.
- Anyone building a production multi-tenant deployment on `ragcore`
  today should **not** wire `GraphRetriever` or `ConceptRetriever` onto
  a multi-tenant `TenantScopedCorpus` until the Tier 2 design above is
  resolved and implemented — doing so today risks real cross-tenant
  data exposure through the graph/concept layers' own merge behavior,
  not just a missed filter.
