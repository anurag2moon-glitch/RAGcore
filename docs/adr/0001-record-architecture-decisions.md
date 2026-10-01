# 1. Record architecture decisions

Date: 2026-10-01

## Status

Accepted

## Context

`ragcore` is growing into a multi-tenant, multi-project retrieval platform
(Phases 1-9 of the long-range build plan: ingestion, embedding/index,
retrieval, GraphRAG, generation/grounding, evaluation/observability,
multi-tenancy, API/SDK/deployment, docs). `CLAUDE.md` already captures a
large number of non-obvious design decisions as prose, written at the time
each decision was made — this works well as a decision *log* but doesn't
give each decision a stable identifier, a single-purpose record, or a
place to mark a later decision as superseding an earlier one.

As the codebase crosses into genuinely new territory (multi-tenancy, a
network-facing API, provider abstractions beyond the first one shipped),
the number of load-bearing decisions that need to survive past the PR that
made them is growing, and some of them (e.g. "SQLite-backed stores first,
Kuzu/Postgres later," "Louvain not Leiden," "no job queue yet") are things a
future contributor could plausibly second-guess or accidentally reverse
without realizing why they were made that way.

## Decision

We will use Architecture Decision Records, as described by Michael Nygard,
for decisions of lasting architectural consequence going forward.

Each ADR is a short Markdown file in `docs/adr/`, numbered sequentially
(`NNNN-title-with-dashes.md`), with sections: Status (Proposed / Accepted /
Deprecated / Superseded by ADR-xxxx), Context, Decision, Consequences. An
ADR is written at the time a decision of lasting consequence is made, not
retroactively reconstructed — except where this audit backfills one for a
decision already visible in the existing codebase/history that had no
record.

This does not replace `CLAUDE.md`. `CLAUDE.md` remains the right place for
dense, implementation-level rationale (why a specific SQL clause, why a
specific cache key shape, a worked example of a subtle bug). ADRs are for
decisions at the level of "which technology/approach, and why, as opposed to
the alternatives" — the kind of thing a Phase 7/8 implementer needs to find
in thirty seconds without reading all 31KB of `CLAUDE.md` end to end.

## Consequences

- Future contributors (including future audit/planning passes) have one
  directory to scan for "why is it built this way" before proposing a
  change that contradicts a settled decision.
- Decisions already made but undocumented as ADRs (e.g. SQLite-first graph
  storage, Louvain over Leiden, no job queue yet, single-provider embedding)
  are not retroactively given numbered ADRs in this pass — the Phase 0
  audit (`docs/ARCHITECTURE.md`) covers them narratively instead, since
  they are existing-code facts, not new decisions made during this audit.
  Backfilling them as formal ADRs is listed as Phase 9 work.
- Adds a small amount of process overhead: a real architectural decision
  now means writing a short file, not just a commit message.
