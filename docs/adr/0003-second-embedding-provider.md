# 3. OpenAI as the second embedding provider, not local sentence-transformers/BGE

Date: 2026-10-01

## Status

Accepted

## Context

`docs/ARCHITECTURE.md`'s gap list (Phase 2, item 8) flagged that the
`Embedder` protocol is real and swappable but only one implementation
ships (`VoyageContextEmbedder`) — "pluggable" was unproven. The original
long-range spec names four candidate providers: OpenAI, Voyage (already
done), Cohere, and local sentence-transformers/BGE.

Local sentence-transformers/BGE pulls in `sentence-transformers` and
`torch` — multi-gigabyte, GPU-stack-adjacent dependencies — and, more
importantly for proving swappability *in this repo's own test suite*,
every existing provider test (`test_voyage_batching.py`) proves batching/
retry/rate-limit logic against an **injected stub client**, never a real
network call or a real downloaded model, so the suite stays fast and
network-free by default (`pytest -m 'not live and not gutenberg'`). A
local embedder's "real" code path is loading actual model weights, which
either needs network access in CI (downloading from the Hugging Face
Hub) or a vendored model checkpoint in the repo — neither fits this
codebase's existing testing philosophy for an embedding provider.

OpenAI's embeddings endpoint, by contrast, is a thin HTTP client exactly
like Voyage's: the same inject-a-stub-client-and-test-batching-logic
pattern applies with zero new testing infrastructure, and the `openai`
PyPI package itself is a lightweight HTTP client with no ML runtime
dependency.

## Decision

Implement `OpenAIEmbedder` (`ragcore/embedding/openai.py`,
`ragcore[openai]` extra) as the second `Embedder`. Local
sentence-transformers/BGE remains in the gap list for whenever a
consumer's actual deployment constraints (no egress to a hosted provider,
or a strict no-third-party-API requirement) make it the needed one — it
is not removed from scope, only sequenced after OpenAI.

Two real behavioral differences from Voyage are accepted and documented
rather than hidden: OpenAI's endpoint has no contextualized-group
embedding (each input embeds independently; `embed_groups()` flattens and
reassembles instead of sending groups jointly) and no document/query
input-type asymmetry (`input_type` is accepted for protocol conformance
and silently has no effect). See the module docstring for the full
rationale on both.

## Consequences

- Proves the `Embedder` protocol is genuinely swappable with a second
  real implementation, using the exact same stub-client test pattern as
  the first, so the test suite's "no network, no native ML deps" property
  is preserved.
- A consumer that specifically needs on-prem/offline embedding still has
  to wait for the deferred local sentence-transformers/BGE implementation
  — this ADR does not close that need, it only explains why it wasn't
  the second provider chosen.
- `OpenAIEmbedder` silently ignoring `input_type` is a real footgun if a
  future reranking/retrieval change starts relying on query/document
  asymmetry mattering for correctness rather than just relevance quality —
  flagged in the module docstring for whoever hits that.
