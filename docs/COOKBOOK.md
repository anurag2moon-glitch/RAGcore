# Cookbook

Phase 9 deliverable, promoted from `scripts/*_demo.py` — every script
below was re-run directly against this exact commit before being
written up here, not described from memory or from its own docstring
alone. Each one is a complete, runnable, working example; none of them
are "part of the library's public API" (see each script's own
docstring) — they're worked examples to read, run, and adapt, not
things a consuming project imports.

Nine of the ten scripts under `scripts/` need **no API keys and no
network** — every LLM-backed component is stood in for by a small,
deterministic fake, the same pattern `tests/fakes.py` and
`tests/pride_prejudice_fixture.py` use for the test suite itself. Only
`scripts/query.py` (optionally) and `scripts/run_eval.py` (optionally,
outside `--demo` mode) touch real vendor APIs.

## Ingest, retrieve, ask — the core pipeline

**`scripts/query.py`** — ad-hoc CLI for querying an already-ingested
corpus.

```
VOYAGE_API_KEY=... python scripts/query.py samples/library.lancedb "your query" [k]
VOYAGE_API_KEY=... ANTHROPIC_API_KEY=... python scripts/query.py samples/library.lancedb "your query" [k] --ask
```

The fastest way to sanity-check retrieval (or, with `--ask`,
generation) quality against a real corpus you've already built — see
`docs/CONFIGURATION.md` for every environment variable this needs.

## GraphRAG

**`scripts/graph_demo.py`** — runs the real `build_graph()` pipeline
against `samples/library.lancedb` (a real, already-ingested *Pride and
Prejudice* corpus checked in for exactly this), substituting a
hand-authored extraction fixture for `ClaudeEntityExtractor`
(`tests/pride_prejudice_fixture.py` — see its own docstring for what's
substituted and why). No API calls at all.

```
python scripts/graph_demo.py
```

What to look for in the output: the merge-outcome assertions at the
end are the most useful part — confirms "Bingley"/"Mr. Bingley" and
"Elizabeth"/"Elizabeth Bennet" correctly merge via
`EmbeddingEntityResolver`, and that "Jane"/"the eldest Miss Bennet"
*doesn't* (the known, documented relational-entity gap — see
`CLAUDE.md`'s "Known limitation" section). This is the concrete,
real-book evidence behind that limitation, not a hypothetical.

## Concepts, communities, and verified generation

**`scripts/concepts_demo.py`** — the full concept layer end to end:
`ingest` → `build_concepts` → `detect_communities` →
`build_community_summaries` → the concept-aware retriever stack
(`MMRReranker(ConceptRetriever(GraphRetriever(VectorRetriever)))`) →
`ask()` with an opt-in `VerifyingGenerator`. Requires the `concepts`
extra (`networkx`) for `detect_communities`; no API calls.

```
python scripts/concepts_demo.py
```

Builds its own tiny synthetic multi-chapter book (written to
`samples/`, gitignored) rather than requiring a pre-built corpus — the
whole thing is self-contained.

## Claims: agreement and conflict across sources

**`scripts/claims_demo.py`** — `ingest` → `build_concepts` →
`build_claims` → `link_claims`, over a small synthetic two-chapter
passage with a *deliberate* agreement and a *deliberate* conflict on
the same concept, so the output has something real to show. No API
calls.

```
python scripts/claims_demo.py
```

What to look for: the printed `author_confidence` ("firm"/"hedged")
per claim and the `conflicts`/`agrees` relation `link_claims()` found
between them — this is the concrete shape of what this layer is for
(see `CLAUDE.md`'s "Claim extraction and linking layer" section for
the full design rationale, including why confidence is judged
independently of agreement).

## Hierarchy and the full seven-layer retriever stack

**`scripts/full_stack_demo.py`** — the complete default stack:

```
MMRReranker(
  CrossEncoderReranker(
    IterativeRetriever(
      HierarchyRetriever(
        ConceptRetriever(
          GraphRetriever(
            HybridRetriever(embedder, store, lexical, corpus))))))))
```

Requires the `concepts` extra; no API calls (every LLM-backed
component, including cross-encoder reranking and sufficiency judging,
is faked).

```
python scripts/full_stack_demo.py
```

What to look for: the printed results mix `kind=passage`,
`kind=hierarchy_summary`, and `kind=community_summary` entries in one
ranked list with comparable scores — the concrete evidence that every
layer's synthetic chunks really do compete on the same scale as real
passages (`ragcore.retrieval.scoring`), not a separately-ranked
side-channel.

## Conversation: multi-turn chat with query rewriting

**`scripts/conversation_demo.py`** — `PersonaChatSession` wrapping one
persona's `Corpus` with a transcript and history-aware query rewriting.
No API calls.

```
python scripts/conversation_demo.py
```

What to look for: turn 3 ("What about when I feel guilty about it?")
only retrieves anything useful because `HistoryAwareQueryRewriter`
resolves it against the transcript first — the script prints both the
raw message and the rewritten query per turn specifically to make that
rewrite step visible rather than implicit.

## Incremental ingestion, snapshots, and rollback

**`scripts/sync_demo.py`** — `sync_corpus()` (diff-based incremental
re-ingestion) plus `create_snapshot()`/`rollback()` end to end: ingest
two books, edit one, re-sync (which auto-snapshots first), then roll
back and confirm both the vector store and the graph store are back to
their pre-sync state. Uses a real on-disk `LanceDBStore` (snapshot/
restore needs LanceDB's native versioning) but fakes every LLM-backed
component. See `docs/RUNBOOK.md` for the full backup/restore
procedure this demonstrates a slice of.

```
python scripts/sync_demo.py
```

## Evaluation

**`scripts/run_eval.py`** — ad-hoc CLI for the `ragcore.eval` harness.

```
python scripts/run_eval.py --demo        # no API keys: synthetic corpus + starter question set
VOYAGE_API_KEY=... python scripts/run_eval.py <corpus-path> <questions.jsonl> [k]
VOYAGE_API_KEY=... ANTHROPIC_API_KEY=... python scripts/run_eval.py <corpus-path> <questions.jsonl> [k] --verify
VOYAGE_API_KEY=... python scripts/run_eval.py <corpus-path> <questions.jsonl> [k] --baseline old_report.json
```

`--demo` is the one worth running first — zero setup, prints a full
`recall@k`/`precision@k`/`ndcg@k`/`mrr`/`hit_rate@k` plus
`latency_p50/95/99` report against a tiny built-in corpus, so you can
see the report's exact shape before pointing it at a real one.
`--verify` additionally scores groundedness (needs `ANTHROPIC_API_KEY`
— it generates an answer and runs `ClaudeVerifier` against it, per
question). `--baseline` is the regression-gate comparison
`.github/workflows/ci.yml`'s own integration suite uses internally.

## Scale and indexing

**`scripts/benchmark_lancedb.py`** — throughput/latency benchmark
across corpus sizes, with no API keys (synthetic random vectors — this
measures `LanceDBStore`/LanceDB itself, not an embedder).

```
python scripts/benchmark_lancedb.py                      # default sizes: 1000,10000
python scripts/benchmark_lancedb.py --sizes 1000,10000,100000
python scripts/benchmark_lancedb.py --dim 1536 --method hnsw_sq
```

Confirmed directly: LanceDB's IVF-PQ index training needs at least 256
rows — a size below that (e.g. `--sizes 200`) raises
`StoreError: ... Not enough rows to train PQ`, not a bug in the script.
Measures, per size: upsert throughput, brute-force query latency,
`create_vector_index()` build time, indexed query latency, and
`recall@10` of the approximate index against brute-force ground truth
— the real quality cost IVF-PQ's approximation trades for its speed,
not just a claim that it's "approximate." The printed `recall@10` is
measured on structureless random vectors, a worst case for IVF-PQ/HNSW
— representative of throughput/latency, not of recall against real
embeddings.

## Reference HTTP/MCP servers and SDKs

Not scripts, but worked examples in their own right — each sibling
package's own README has the exact run/test commands:

- `server/README.md` — the reference FastAPI server.
- `mcp-server/README.md` — the MCP server.
- `sdk/python/README.md`, `sdk/typescript/README.md` — the two client
  SDKs, each with a complete usage example.

## Fetching real test fixtures

**`scripts/fetch_gutenberg.py`** — downloads the real Project
Gutenberg books the opt-in `@pytest.mark.gutenberg` integration-suite
layer uses, into a local gitignored cache. Idempotent; run once before
`pytest -m gutenberg`.

```
python scripts/fetch_gutenberg.py
```
