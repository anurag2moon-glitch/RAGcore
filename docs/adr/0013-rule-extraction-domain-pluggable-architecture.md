# 13. Rule extraction/base: generic core in ragcore, domains as sibling packages

Date: 2026-10-07

## Status

Accepted

## Context

Phase 10 asks for a rule-extraction and rule-base system that is
"domain-pluggable" from day one, shipping a first concrete domain (Lal
Kitab astrology) without baking that domain into the core. This is
exactly the "generic vs. domain-specific" test this project already
applies to every component (see `CLAUDE.md`'s framing of `ragcore` as
the shared base for more than one downstream product). Before writing
any code, an audit of the existing closest analogue —
`ragcore/claims/` (atomic claim extraction + agree/conflict/extends
linking) and `ragcore/graph/` (entity/relation extraction) — found
substantial, directly reusable machinery:

- **Content-addressed ids** (`ragcore/claims/claude.py::_claim_id` —
  `sha256(source_id|chunk_id|normalized_text)`) — the same scheme this
  ADR's rule schema reuses for `rule_id`, giving idempotent re-extraction
  for free (`docs/adr/0010-content-addressed-identity.md`).
- **The embedding-shortlist batching pattern**
  (`ragcore/claims/linking.py::_candidate_batches`) — bounds a
  corpus-global pairwise comparison (claim-vs-claim there, rule-vs-rule
  here) to sub-quadratic cost deterministically, avoiding the lossy
  random sampling `ragcore.concepts.communities` uses elsewhere
  (rejected there specifically because it would make an unchanged
  corpus produce a different relation set on every run — the same
  nondeterminism a conflict/precedence detector must avoid).
- **The two-pass-extraction-plus-independent-verifier pattern**
  (`ragcore/verification/claude.py`'s `ClaudeVerifier`: a separate
  `messages.parse()` call checks whether each claim is *actually*
  supported by its cited source, never trusting the extracting model's
  own self-assessment) — directly what Phase 10 §2's "Pass 3:
  independent verifier model checks that the verbatim span actually
  supports every condition and the effect" asks for.
- **Per-chapter checkpointing with a content-hash fingerprint**
  (every `build_*()` pass in this codebase) — directly reusable for
  incremental rule extraction (§2's "re-extract only changed chunks").
- **Immutable-version-on-edit, not merge-in-place** — new territory:
  every existing layer (graph, concepts, claims) accumulates
  provenance into one row per content-addressed id
  (`docs/adr/0011-graph-merge-not-replace.md`). A rule under review
  (§4: "Edits create new versions and never mutate in place") needs
  the *opposite* discipline — an edit must never overwrite what a
  reviewer rejected or what an earlier version asserted, since the
  audit trail is the whole point. This is a real, new storage pattern
  this layer introduces, not a reuse of the existing one.

## Decision

**Generic rule-extraction/rule-base infrastructure lives in
`ragcore/rules/`** — the schema, the domain-pack `Protocol`, the
extraction pipeline, the consistency/conflict-detection layer, the
review-workflow store, the serving/evaluator, and the eval harness.
None of it mentions Lal Kitab, planets, or houses.

**Each concrete domain (Lal Kitab first, BPHS or others later) is a
separate sibling package**, matching the exact pattern
`docs/adr/0008-api-server-as-sibling-package.md` already established
for `server/`, `mcp-server/`, `sdk/python/`, `sdk/typescript/`: its own
directory (`domains/lal_kitab/`), its own `pyproject.toml`, depending
on `ragcore` and implementing `ragcore.rules.domain`'s `DomainPack`
Protocol (ontology/alias normalization, extraction prompts, fact
types the evaluator understands). `ragcore`'s own `pyproject.toml`
gains no Lal-Kitab-specific dependency or vocabulary, the same
discipline that's kept `ragcore/` itself pure of FastAPI/MCP/SDK
concerns since Phase 8.

A `DomainPack` is *inert data plus pure functions* (ontology mappings,
prompt templates, a fact-typing schema) — it is never itself an
extraction pipeline; the generic pipeline in `ragcore/rules/` drives
extraction, calling into the pack only for domain knowledge. This is
what makes "add BPHS without core changes" (§7) actually true rather
than aspirational: a new domain pack never needs to touch
`ragcore/rules/`'s pipeline, consistency, review, or serving code.

## Consequences

- Rule schema, pipeline, consistency layer, and review workflow are
  built and tested once, generically, against a synthetic test domain
  (no real Lal Kitab content needed to test the *infrastructure* —
  matching this whole codebase's established discipline of testing
  against `tests/fakes.py`-style doubles rather than live content
  wherever the thing under test is domain-agnostic).
- The Lal Kitab pack is a thin, separately-versioned package a future
  BPHS pack doesn't have to touch or even import.
- `ragcore.verification.ClaudeVerifier`'s pattern is adapted, not
  reused as-is: a rule's "claims" are structured (conditions + effect),
  not decomposed prose, so the verification pass checks structured
  support against a verbatim span directly, in one call, rather than
  decompose-then-check in two.
- The new immutable-version-on-edit storage discipline (review
  workflow) is **not** retrofitted onto `ragcore/claims/` or
  `ragcore/graph/` — those layers' existing merge-not-replace contract
  stays exactly as `docs/adr/0011` describes it; this ADR adds a
  second, different pattern alongside it for a genuinely different
  need (human review and auditability vs. automatic provenance
  accumulation), not a replacement.
