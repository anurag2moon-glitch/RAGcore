# 6. tenant_id is a foundation field, not enforced isolation

Date: 2026-10-06

## Status

Accepted

## Context

`docs/ARCHITECTURE.md` names Phase 7 (multi-tenancy) "the next major
phase boundary, not an extension of existing code," and its own gap list
says to "start with `tenant_id` propagation through `ChunkMeta` and every
store schema, since every other Phase-7 item (quotas, isolation,
metering) depends on that field existing first." The full Phase 7 scope
named in the original spec — API keys, JWT/OAuth, RBAC, per-tenant
quotas, rate limiting, usage metering, semantic/result caching with
tenant-scoped invalidation, data export/hard-delete/retention policies,
encryption at rest and in transit, audit logs, PII detection,
prompt-injection defenses, per-document licensing flags — is realistically
weeks of focused engineering for a dedicated team, not something to
attempt in one pass without risking shallow, undertested work across all
of it at once.

## Decision

This pass does exactly the foundation step the gap list names, and
nothing beyond it: `ChunkMeta.tenant_id: str | None = None` (opt-in,
additive — every existing caller that never passes `tenant_id=...` sees
`None` on every chunk, identical to before this field existed), wired
through `Corpus.ingest()`/`ingest_many()`'s new `tenant_id=...` parameter,
and `LanceDBStore`'s schema/filterable-fields (so `corpus.retrieve(...,
filters={"tenant_id": "..."})` works end to end for the primary vector
store).

This field makes tenant-scoped filtering *possible*; it does **not**
itself enforce isolation. There is no guard anywhere in this pass against
a caller forgetting the `tenant_id` filter on a `retrieve()`/`ask()` call
and silently reading across tenants — that enforcement (e.g. a
`Corpus` wrapper that always injects the filter, or a server-layer
middleware that never lets a request omit it) is real, necessary,
follow-on work this pass deliberately does not claim to have done. This
mirrors a limitation already documented elsewhere in this codebase for
an analogous reason: CLAUDE.md's claims-layer section states "one
`Corpus`/`ClaimStore` per persona" as the product's assumption, not
something `ragcore` enforces mechanically — the same shape of gap, now
named explicitly for `tenant_id` rather than left implicit.

Every other store (`SqliteGraphStore`/`KuzuGraphStore`,
`SqliteConceptStore`, `SqliteClaimStore`, `Fts5LexicalStore`, etc.) does
**not** get a `tenant_id` column in this pass — only `LanceDBStore`, the
primary vector store a `Corpus` always has. Extending the same pattern to
the graph/concepts/claims/lexical stores, and to actual enforcement, is
explicitly left as follow-on Phase 7 work, not silently assumed done.

## Consequences

- Zero behavior change for every existing caller, consistent with every
  other opt-in addition this project has made (`checkpoint_store`,
  `embedding_cache`, `context_max_tokens`, `gleaning_rounds`,
  `enable_prompt_caching`) — `tenant_id` fits the same proven shape rather
  than inventing a new one.
- A caller who *does* want real tenant isolation today must build the
  enforcement layer themselves on top of this field (always pass the
  filter, never trust an unfiltered query) — this pass gives them the
  mechanism, not the guarantee. Documented here so that gap is visible,
  not discovered the hard way.
- The remaining Phase 7 scope (auth, RBAC, quotas, billing-ready
  metering, caching, data lifecycle, encryption, PII/prompt-injection
  defenses, licensing flags, and `tenant_id` on the other stores) is
  tracked in `docs/ARCHITECTURE.md`'s gap list as open, not folded into
  "done."
