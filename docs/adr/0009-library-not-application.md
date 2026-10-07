# 9. ragcore is a library, not an application

Date: 2026-10-07 (backfilled — see "Status" below)

## Status

Accepted. **Backfilled**: this decision was made at the project's
founding, long before this repository adopted the ADR practice
(`docs/adr/0001-record-architecture-decisions.md`). It is recorded now,
retroactively, because `docs/ARCHITECTURE.md`'s Phase 9 gap list calls
for exactly this ("ADR backfill for major decisions already made"), and
because it is the single decision every later ADR in this directory
either depends on or has to actively argue around (`0008`'s API server
living in a sibling `server/` directory specifically to avoid violating
it is the clearest example).

## Context

`ragcore` exists to serve more than one downstream product from one
shared retrieval base — originally Mimisi (a philosopher-chat product)
and Vastu Shastra (a rule-engine product), with an eventual
multi-tenant hosted product as a further consumer down the line
(`CLAUDE.md`'s own framing). A shared base used by several independent
products has a different obligation than an application: it cannot
assume *how* it will be run, only that it will be imported and driven
by code the base doesn't control.

If `ragcore` had grown a CLI, a server, or any module-level mutable
global state, every consumer would inherit those choices whether or
not they wanted them — a CLI entrypoint drags in argument-parsing
dependencies for a consumer that only ever calls the library
programmatically; a bundled server forces every consumer to either run
it or explain why they don't; global state (a module-level singleton
client, a process-wide cache) silently couples two unrelated consumers
running in the same process, or breaks when a consuming application
itself wants multiple independent `Corpus` instances (one per tenant,
one per test — see `ragcore.tenancy.TenantScopedCorpus`, which is only
possible to build correctly because `Corpus` instances are already
fully independent of each other).

## Decision

`ragcore` carries exactly one, explicit rule: **library, not
application — no CLI, no server, no global state.** Every capability is
a class or a function a consumer constructs and composes; nothing in
the package reads `sys.argv`, binds a socket, or holds state outside an
instance a consumer explicitly created. Every swappable piece (embedder,
vector store, generator, chunker, retriever layer, cache, detector) is
a `Protocol`, and every optional capability is a decorator or a
standalone pass over an already-constructed `Corpus` — never a
side-effecting import or an auto-wired default a consumer has to
actively opt out of.

The one directly observable consequence this session: when Phase 8
needed a real HTTP server and a real MCP server to exist, both were
built as separate sibling packages (`server/`, `mcp-server/`) each with
their *own* `pyproject.toml`, rather than inside `ragcore/` itself (see
`docs/adr/0008-api-server-as-sibling-package.md`). Neither `fastapi`,
`uvicorn`, nor `mcp` was ever added to `ragcore`'s own dependency list.
A consumer who only ever calls `ragcore.Corpus` directly, never runs
either server, pays nothing for either existing.

## Consequences

- Every new capability added to this codebase (across this entire
  session: `ragcore.tenancy`, `ragcore.privacy`, `ragcore.injection`,
  `ragcore.caching`, and everything before them) had to pass the test
  "does this reasonably belong behind an interface a consumer
  constructs," not "is this convenient to wire up by default." Several
  ADRs in this directory are a direct record of staying on the right
  side of that line under real pressure to cut a corner (`0006`'s
  `TenantScopedCorpus` enforcing nothing by itself without a consumer
  explicitly using it; `0007` refusing to ship a tenant_id column for
  the graph/concept stores that would have looked done without being
  safe).
- The cost: nothing in `ragcore` can give a consumer a complete,
  working system out of the box — every consumer (Mimisi, Vastu
  Shastra, the reference `server/`) has to do real integration work to
  turn the library into something runnable. This is accepted as the
  correct tradeoff for a base meant to serve more than one product with
  different needs, not treated as a gap to eventually close by growing
  a default application inside the library itself.
