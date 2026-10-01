# ragcore — Architecture Audit (Phase 0)

Audit date: 2026-10-01. Scope: everything under `ragcore/`, `tests/`, `scripts/`,
`samples/`, plus `CLAUDE.md`, `README.md`, and `docs/INTEGRATION_TEST_FINDINGS.md`.
Read-only pass — no code changed. Written against the 9-phase target spec
(ingestion → embedding/index → retrieval → GraphRAG → generation/grounding →
eval/observability → multi-tenancy/platform → API/SDK/deployment → docs).

## 0. What this codebase is, today

`ragcore` is a **library, not an application**: no CLI, no server, no global
state. A consuming project does `from ragcore import Corpus`, builds a
`Corpus`, and drives it directly. It is already the shared base for two
downstream products (per project context: Mimisi, a philosopher-chat app, and
Vastu Shastra, a rule engine) — see §4 for how well the current code respects
that boundary.

Total (updated after Phase 1, 2026-10-01): ~12,700 lines across
`ragcore/`, 25 submodules (added `ingestion/`), ~85 test files (336 pass
by default; 10 more opt-in via `pytest -m "live or gutenberg"`).
Zero `TODO`/`FIXME`/`XXX` markers anywhere in the package — every
`except Exception` is deliberate and `# noqa`-annotated with a reason. This
is a carefully, incrementally built codebase, not a prototype; **Phase 1-6
of the target spec are substantially (not fully) covered already**, and
Phase 7-8 do not exist at all. See §2 for the precise breakdown.

`CLAUDE.md` (31KB) already functions as a living, detailed design-decision
log — far more detailed than typical repo docs — covering nearly every
non-obvious choice in the codebase. This audit does not duplicate it;
read `CLAUDE.md` itself for the full reasoning behind any item below marked
"see CLAUDE.md". **One stale reference found**: both `README.md` and
`CLAUDE.md` point to `~/.claude/plans/i-m-building-a-reusable-cosmic-quasar.md`
as "the full design doc" — that file does not exist anywhere on disk. Either
it was moved/deleted, or the pointer was never updated after a path change.
Not fixed in this pass (read-only); flagged for whoever owns docs next.

## 1. Module inventory

Every module follows the same shape: a `Protocol` in `base.py` (or `types.py`
for data-only packages), a default implementation, a `pipeline.py`
orchestration function where relevant, and a store. Swappability is real —
`Corpus` only ever holds `VectorStore`/`Embedder`/`Chunker`/`Loader`/
`Generator` protocols, and every add-on layer attaches through the
`corpus.retriever` / `corpus.generator` seams without touching `Corpus`.

| Module | Purpose | Default impl(s) | Tested |
|---|---|---|---|
| `types.py` / `errors.py` | Core frozen dataclasses (`Chunk`, `ChunkMeta`, `Document`, `Retrieved`, ...); typed exception hierarchy | — | Indirectly, via every other module's tests |
| `loaders/` | File → `Document` | `pdf.py` (PyMuPDF, text + OCR fallback via injectable `ocr_fn`), `epub.py`, `docx.py`, `text.py` (txt/md), `html.py` | `test_loaders_structure.py`, `test_pdf_ocr.py` — incl. malformed-input handling |
| `chunking/` | `Document` → `ChunkGroup`s | `StructuralChunker` (chapter-aware, token-budget packing, overlap, paragraph/sentence/char-level oversized splitting); `SemanticChunker` (embedding-boundary breakpoints) | `test_chunk_id_stability.py`, `test_semantic_chunker.py`, `test_page_hash.py` + integration suite |
| `ingestion/` | Opt-in mid-`Corpus.ingest()` checkpointing + `ingest_many` dead-letter handling | `IngestCheckpointStore` protocol, `SqliteIngestCheckpointStore` | `test_ingest_checkpoint_store.py`, `test_ingest_checkpointing.py` |
| `embedding/` | Text → vectors | `VoyageContextEmbedder` (contextualized), `OpenAIEmbedder` (flattens groups, no contextualization — see its module docstring); both batched/retried; `cost.UsageTracker` (`.usage`) and `cache.SqliteEmbeddingCache` (opt-in via `Corpus(embedding_cache=...)`) | `test_voyage_batching.py`, `test_openai_embedder.py`, `test_embedding_cost.py`, `test_embedding_cache*.py`; live tests opt-in |
| `stores/` | Vector persistence + similarity query + filter DSL | `LanceDBStore` (embedded, on-disk, filter allowlist, snapshot/restore via native tags, `create_vector_index`/`compact`/`index_stats`) | `test_store_swap.py`, `test_lancedb_snapshot.py`, `test_lancedb_index_lifecycle.py`, integration concurrency/scale suite |
| `retrieval/` | Query → ranked `Retrieved[]`, composable decorator chain | `VectorRetriever`, `HybridRetriever` (RRF + BM25 via FTS5), `GraphRetriever`, `ConceptRetriever`, `HierarchyRetriever`, `IterativeRetriever` (multi-hop), `CrossEncoderReranker`, `MMRReranker`, `QueryExpansionRetriever` (HyDE/multi-query/decomposition), `RoutedRetriever` (per-query mode routing); `fusion.rrf_merge` shared by the last two; `stack.default_stack()` wires the always-on default chain | One test file per layer + `test_default_stack.py` + integration load/query tests |
| `lexical/` | BM25 full-text index | `Fts5LexicalStore` (stdlib SQLite FTS5, zero new deps) | `test_lexical_fts5.py`, `test_lexical_pipeline.py` |
| `graph/` | Entity/relation extraction + GraphRAG store | `ClaudeEntityExtractor`, `SqliteGraphStore` (WAL, locked, merge-not-replace), `EmbeddingEntityResolver` | ~19 tests incl. concurrency, cross-book resolution, a real P&P fixture |
| `concepts/` | Abstract concept/theme extraction + community detection + summaries | `ClaudeConceptExtractor`, `SqliteConceptStore`, Louvain via `networkx` (`concepts` extra), `EmbeddingConceptMapper`/`ClaudeConceptMapper` | `test_communities.py`, `test_concept_*.py` (extraction/removal/resume/retrieval/store) |
| `contextual/` | Per-chunk situating blurb before re-embedding | `ClaudeContextWriter`, `SqliteContextStore` | `test_contextual_embeddings.py`, `test_contextual_resume.py` |
| `hierarchy/` | chunk→section→chapter→book summarization | `ClaudeHierarchySummarizer`, `SqliteHierarchyStore` | `test_hierarchy*.py` (5 files) |
| `claims/` | Atomic claim extraction + agree/conflict/extends linking (storage only, no retriever) | `ClaudeClaimExtractor`, `ClaudeClaimLinker`, `SqliteClaimStore` | `test_claims.py`, `test_claim_linking.py`, `test_claim_removal.py` |
| `conversation/` | Session transcript + history-aware query rewriting on top of `Corpus.ask()` | `InMemoryTranscriptStore`, `HistoryAwareQueryRewriter`, `PersonaChatSession` | `test_conversation_chat_session.py`, `test_conversation_transcript.py` |
| `generation/` | Context + query → `Answer` | `ClaudeGenerator`, `context_packer.pack_context` (opt-in token-budget packing, dedupe, neighbor expansion, document ordering) | `test_generation.py`, `test_context_packer.py`, `test_claude_generator_context_packing.py`; live test opt-in |
| `verification/` | Post-hoc per-claim faithfulness check | `ClaudeVerifier`, `VerifyingGenerator` (decorator) | `test_verification.py` |
| `eval/` | General-purpose retrieval/groundedness eval harness for a *consumer's* corpus | `run_eval()`, `GoldQuestion` JSONL format, recall/precision/nDCG/MRR + optional groundedness | `test_eval_*.py` (4 files); distinct from `tests/integration/eval/`, ragcore's own internal regression harness |
| `versioning/` | Incremental re-ingest (diff/sync), snapshots, rollback | `IngestionManifest`, `plan_sync`/`sync_corpus`, `LanceDBStore.snapshot()`, SQLite online-backup | `test_ingestion_manifest.py`, `test_sync_corpus.py`, `test_snapshot_rollback.py` |
| `reranking/` | Cross-encoder rerank scoring | `VoyageReranker` (reuses Voyage API, zero new dep) | `test_voyage_reranker.py`, `test_cross_encoder.py` |

`tests/integration/` (~2,700 lines, real components + deterministic mock
embedder/extractor, zero network) is a separate, production-scale regression
suite: 8 synthetic books (+ 8 real Gutenberg books opt-in) covering multi-book
ingestion, cross-book entity resolution, concurrent writers, failure
injection, query load, and a committed eval baseline
(`recall@10=1.0, nDCG@10≈0.908, MRR=0.875` at full-library scale). This is a
genuinely unusual amount of rigor for a pre-Phase-7 library and should be
preserved and extended, not replaced, as Phase 1+ work lands.

## 2. Coverage against the 9-phase target spec

Legend: ✅ covered · 🟡 partial · ⬜ missing (0%)

### Phase 1 — Ingestion pipeline — **complete as of 2026-10-01**

Every item below is now ✅ except the one explicitly deferred to Phase 7
(tenant/edition/language/ingest_version fields — deferred so they aren't
designed twice). See commits `c350ff5`, `e59d693`, `f7d55f1`, `770f38e`,
`6596b95`.

- ✅ Per-stage checkpointing for *ingestion itself*, opt-in via
  `Corpus(checkpoint_store=...)` (`ragcore.ingestion`,
  `docs/adr/0002-ingest-checkpointing-opt-in.md`): resumable per-`ChunkGroup`,
  with dead-letter handling in `ingest_many(max_retries=...)`. Left opt-in
  (not always-on) deliberately — see the ADR. A distributed job queue
  (multiple workers/processes pulling from a shared queue, as opposed to
  this single-process, in-library resumability) remains Phase 7/8 work —
  this closes the *checkpointing* half of the original gap, not the
  *queue* half, which belongs with the platform/deployment phases.
- ✅ Resumable/idempotent, including mid-file (previously: only
  whole-file/cross-file via `source_id`/`sync_corpus`).
- ✅ Content-hash dedupe at file, page (`Block.page_hash`, PdfLoader only,
  propagated to `ChunkMeta.extra["page_hash"]` for a chunk that doesn't
  straddle a page boundary), and chunk level.
- ✅ Parsers: PDF, EPUB, DOCX, TXT, MD, HTML (`loaders/html.py`) all behind
  the one `Loader` interface; PDF OCR fallback for scanned pages
  (`PdfLoader(ocr=True)`, pytesseract via an injectable `ocr_fn` seam,
  `ragcore[ocr]` extra).
- ✅ Book-aware structure (unchanged from before this pass).
- ✅ Hierarchical chunking (unchanged).
- ✅ Layout-aware/semantic chunking, pluggable, configurable per document
  type — now two real strategies: `StructuralChunker` (default,
  token-budget packing) and `SemanticChunker` (embedding-boundary
  breakpoints, `ragcore/chunking/semantic.py`), both behind the same
  `Chunker` protocol with no changes to it.
- ✅ Contextual chunk enrichment (unchanged).
- 🟡 Rich metadata schema — `tenant_id`/`edition`/`language`/
  `ingest_version` remain deliberately deferred to Phase 7 (see Phase 1
  gap list item 5 below, now the only open item in this phase).
- ✅ Multi-file/folder batch upload, progress events, partial-failure
  reporting, re-ingest diffing (unchanged).

### Phase 2 — Embedding and index layer — **functionally complete as of 2026-10-01**
- ✅ Provider abstraction — two providers now ship: `VoyageContextEmbedder`
  and `OpenAIEmbedder` (`ragcore/embedding/openai.py`,
  `docs/adr/0003-second-embedding-provider.md` for why OpenAI over local
  sentence-transformers/BGE as the second one — that remains a gap item,
  just sequenced later). Batching/rate-limit/retry — ✅ for both.
  **Cost tracking — ✅** (`ragcore.embedding.cost.UsageTracker`, wired into
  both providers' `self.usage`; exact token/request counts always, a
  dollar estimate only when a price is supplied or known for the model —
  never a guessed number presented as fact).
- ✅ Embedding cache keyed by `(content_hash, model_name, dim)` —
  `ragcore.embedding.cache` (`EmbeddingCache` protocol,
  `SqliteEmbeddingCache` default), opt-in via `Corpus(embedding_cache=...)`.
  Caches at whole-`ChunkGroup` granularity (not per-chunk) to preserve
  contextualized-embedding integrity — a group is skipped only when every
  chunk in it is already cached. Zero-downtime re-embedding migration
  falls out of the key shape for free: a model/dim change is a guaranteed
  cache miss, so the old model's entries are undisturbed and still
  servable while the new model's entries populate alongside them.
- 🟡 Hybrid indexing: dense ✅ (LanceDB), BM25 ✅ (`lexical/` FTS5), full-text
  — same as BM25 here; **SPLADE — ⬜**
- 🟡 Index lifecycle management — IVF-PQ/HNSW tuning and compaction: ✅
  (`LanceDBStore.create_vector_index()`/`compact()`/`index_stats()`);
  benchmarks: ✅ at the sizes actually run (`scripts/benchmark_lancedb.py`,
  numbers in `docs/BENCHMARKS.md`, up to 50k chunks — 1M/10M are real,
  runnable-but-not-yet-run gaps, left for real target hardware); per-tenant
  table/partition strategy — ⬜, still Phase 7. Versioned tables / rollback
  — 🟡 (unchanged: LanceDB native tag/restore is wired for snapshots, not
  for index-tuning rollback).
- 🟡 S3-compatible object storage for raw files — ✅
  (`ragcore.raw_files.S3RawFileStore`, tested against `moto`'s in-process
  S3 mock). Postgres for metadata/jobs — 🟡: one protocol
  (`IngestCheckpointStore`) has a real Postgres implementation
  (`ragcore.ingestion.PostgresIngestCheckpointStore`), but unlike every
  other store in this codebase it has **not been run against a live
  Postgres server** — this environment has no Docker, no package manager,
  and no Postgres binary available to test against (see
  `docs/adr/0005-postgres-store-untested-against-live-server.md`); treat
  it as implemented-but-unverified-by-execution until run once via
  `pytest -m postgres`. Every other SQLite-backed store (graph, concepts,
  claims, contextual, hierarchy) still has no Postgres counterpart.
  Tenancy — ⬜, still Phase 7.

### Phase 3 — Retrieval engine — **complete as of 2026-10-01**
- ✅ Hybrid retrieval with RRF, tunable — `HybridRetriever` (unchanged).
- ✅ Query understanding — standalone retrieval-layer primitive now exists
  alongside the conversation layer's history-aware rewriting:
  `ragcore.retrieval.query_expansion` (`ClaudeQueryExpander` — one call
  produces a HyDE passage, alternative phrasings, and sub-questions
  together; `QueryExpansionRetriever` fuses the original query's and
  every variant's results via RRF).
- ✅ Cross-encoder reranking, MMR diversity, filters — unchanged.
- ✅ Query router — `ragcore.retrieval.routing` (`ClaudeQueryRouter` +
  `RoutedRetriever`): picks vector/keyword/graph_local/graph_global (or
  any corpus-specific named modes) per query, fusing a combination via
  the same RRF helper. An alternative composition point alongside
  `default_stack()`, not a replacement for it.
- ✅ Agentic multi-hop retrieval — unchanged (`IterativeRetriever`).
- ✅ Context packing — `ragcore.generation.context_packer.pack_context`:
  dedupe, token budgeting, document-order presentation, and opt-in
  parent/neighbor expansion (via `Chunk.chunk_id`'s positional scheme,
  no hard dependency on `ragcore.hierarchy`). Wired into `ClaudeGenerator`
  via opt-in `context_max_tokens=...` — unset, `generate()` is unchanged.

New shared primitive from this phase: `ragcore.retrieval.fusion.rrf_merge`,
extracted once a second caller (the query router) needed the identical
RRF-merge-by-label logic `QueryExpansionRetriever` already had inline.

### Phase 4 — GraphRAG
- 🟡 Entity/relation extraction with typed schemas — `graph/` extracts
  entities/relations; schema is generic (type + name), not the specific
  book-tuned taxonomy the spec names (characters/concepts/people/places/
  works/events/themes/claims) — though `concepts/` (themes) and `claims/`
  (claims) exist as **separate layers**, not unified into one graph schema.
  Gleaning passes (multi-round extraction) — ⬜, not found
- ✅ Entity resolution/canonicalization — `EmbeddingEntityResolver`
  (fuzzy + embedding match), content-addressed ids, alias merging,
  cross-document linking proven at scale in integration tests. One
  documented, accepted limitation: relationally-described entities with no
  shared vocabulary (see CLAUDE.md "Known limitation")
- 🟡 Graph storage behind an interface — ✅ interface (`GraphStore` protocol);
  **only a SQLite implementation ships — no Kuzu, no Neo4j/Postgres
  adapter** (spec wants Kuzu first)
- 🟡 Hierarchical community detection — **Louvain via `networkx`, not
  Leiden** as the spec specifies; community reports/summaries at one level
  only (concepts/communities), not explicitly multi-level
- ⬜ Search modes: local/global/DRIFT-style hybrid as named, selectable
  modes — doesn't exist as such; `GraphRetriever` (local-ish, entity
  neighborhood) and `ConceptRetriever` (community-summary-ish, global-ish)
  exist as separate always-on layers, not a mode selector
- ✅ Incremental graph updates, no full rebuild — `sync_corpus` +
  `remove_chunk_ids`/`remove_source` on every graph-adjacent store
- ✅ Provenance on every node/edge back to chunk — content-addressed ids +
  `source_chunk_ids` unioned on every merge

### Phase 5 — Generation and grounding
- 🟡 Inline citations at chunk level — ✅ (`Answer.citations`, numbered
  `[n]` excerpts in the prompt); **page/chunk clickable spans** — the data
  (`page_start`/`page_end`) is there, but nothing renders "clickable spans";
  this is partly a frontend concern outside a library's scope
- ✅ Faithfulness verification — `ClaudeVerifier`/`VerifyingGenerator`,
  per-claim verdicts, `status="supported_by_synthesis"` distinction for
  community-summary-sourced claims
- ⬜ Streaming (SSE) — not applicable yet, no server exists; prompt caching
  — not configured on the `anthropic.Anthropic()` client calls; model
  routing (cheap model for rewriting/extraction, strong model for synthesis)
  — ⬜, every Claude-backed component takes its own `model` param but nothing
  centrally routes cheap-vs-strong; fallbacks — 🟡 (retry-with-backoff only,
  no cross-provider fallback)
- 🟡 Abstention when evidence is insufficient — the default system prompt
  instructs the model to say so explicitly; no mechanical enforcement (it's
  prompt-level, not a checked behavior)

### Phase 6 — Evaluation and observability
- 🟡 Eval harness — ✅ for a consumer's own corpus (`ragcore/eval/`:
  recall@k/MRR/nDCG/precision/groundedness); **synthetic QA generation from
  books — ⬜** (no generator for gold questions, only a manual authoring
  helper `find_gold_passage`); answer-relevance metric — ⬜ (only
  groundedness, not relevance); latency/cost per query — ⬜, not tracked
  anywhere
- ⬜ CI regression gate — the committed `eval_baseline.json` is a snapshot,
  not wired into CI to block merges on regression; no CI config exists in
  this repo at all (no `.github/workflows/`)
- ⬜ A/B framework for chunking/embedding/reranker/prompt variants — doesn't
  exist
- ⬜ OpenTelemetry tracing, structured logs, per-tenant dashboards,
  Langfuse-style trace viewing — none of this exists; **zero logging calls
  found anywhere in `ragcore/`** (errors raise typed exceptions but nothing
  is logged in the library itself, which is defensible for a library but
  means a consumer has no trace surface to hook into yet)

### Phase 7 — Multi-tenancy and platform
⬜ **Entirely absent.** No `tenant_id` anywhere in `types.py` or any store
schema. No API keys, JWT/OAuth, RBAC, quotas, rate limiting, usage metering.
No semantic/result cache. No export/hard-delete/retention policy beyond
`Corpus.remove(source_id)` (which deletes vectors but the design doesn't
yet reach every graph/concept/claim/hierarchy/context store uniformly as a
single "delete everything for this tenant" operation — `sync_corpus`'s
prune logic is the closest thing and operates per-source, not per-tenant).
No encryption-at-rest config, audit logs, PII detection, prompt-injection
defenses for ingested content, or copyright/licensing flag per document.
This is expected — see §4, this is explicitly the next product-layer
concern, deliberately not pulled into the shared base prematurely.

### Phase 8 — API, SDKs, deployment
⬜ **Entirely absent.** No FastAPI/any HTTP server, no OpenAPI spec, no
gRPC, no generated SDKs, no MCP server, no Docker/Helm/k8s manifests, no
Redis, no feature flags, no load tests, no backup/restore runbook (though
the *mechanism* for backup/restore exists per-store — `backup_to`/
`restore_from_backup`/snapshot/restore — just no runbook describing it).

### Phase 9 — Docs
🟡 `CLAUDE.md` + `README.md` are unusually thorough as *decision logs*, but
there is **no ADR directory before this pass**, no Mermaid system diagram,
no formal data-model reference (beyond `types.py` docstrings), no
configuration reference, no runbook, no cookbook of worked per-downstream-
project examples (though `scripts/*_demo.py` are close to cookbook material
already).

## 3. What's fragile

1. ~~Unwrapped backend exceptions throughout the storage layer~~
   **Resolved in Phase 2** (gap list item 7): `ragcore.errors.wrap_backend_errors`
   is now applied to every public method of every storage-layer class
   (`SqliteGraphStore`, `SqliteConceptStore`, `SqliteClaimStore`,
   `SqliteContextStore`, `SqliteHierarchyStore`, `Fts5LexicalStore`,
   `IngestionManifest`, `LanceDBStore`, plus `SqliteIngestCheckpointStore`
   and `SqliteEmbeddingCache` added since the original audit) — see
   `docs/adr/0004-wrap-backend-exceptions-broadly.md` for the broad-catch
   rationale. `errors.py`'s "All library-raised exceptions derive from
   RagcoreError" contract is now actually true throughout the storage
   layer, confirmed by
   `tests/integration/test_failure_injection.py::test_sqlite_graph_wal_sidecar_loss_raises_a_typed_store_error`
   (previously asserted the *old*, unwrapped behavior; now asserts the fix).
2. ~~No stage-level resumability inside a single `Corpus.ingest()` call.~~
   **Resolved in Phase 1** via opt-in `checkpoint_store=...`
   (`ragcore.ingestion`) — see `docs/adr/0002`.
3. **Everything is single-process, single-machine.** `SqliteGraphStore`
   etc. are hardened for concurrent threads/processes against one file
   (verified in integration tests), but there is no distributed-queue or
   multi-node story — a real blocker the moment "thousands of users" load
   requires horizontal scaling (Phase 7/8 work, not yet started).
4. **`IngestionManifest.backup_to()` has no matching restore-from-backup
   call symmetrical to the `.sqlite3` stores' `restore_from_backup`** —
   worth checking when building the Phase 7 backup runbook (not fully
   verified in this pass; flagged for the Phase 1/7 implementer to confirm).
5. **Claims layer has a documented, *unenforced* invariant**: "one
   `Corpus`/`ClaimStore` per persona" — nothing in `ragcore` stops two
   personas from sharing a store, which would silently cross-contaminate
   `link_claims()`/`detect_communities()`. Fine as a library (the guard is
   correctly left to the product layer per CLAUDE.md's own reasoning), but
   worth a loud callout in any multi-tenant platform spec that builds on
   this, since "no cross-tenant leakage" becomes a hosting-layer
   responsibility, not a ragcore one, under the current design.
6. **A known, deliberately-unfixed quality interaction**: attaching a
   relation hint to an entity for disambiguation can *dilute* that same
   entity's own alias-merge similarity elsewhere (see
   `INTEGRATION_TEST_FINDINGS.md`). Real, not hypothetical, low urgency
   until a real target corpus shows it costing retrieval quality.
7. ~~No page-level dedupe~~ **Resolved in Phase 1**
   (`Block.page_hash`/`ChunkMeta.extra["page_hash"]`).
8. ~~`SourceType` is a closed 5-value `Literal`~~ **Resolved in Phase 1**:
   now six values (added `"html"`); confirmed via grep that no call site
   pattern-matches on it exhaustively, so this was additive-only in
   practice, not the breaking change this item originally warned about.
9. ~~`StructuralChunker` is the only chunking strategy~~ **Resolved in
   Phase 1**: `SemanticChunker` is a second, real implementation.
10. **New from Phase 1**: `Chunk.chunk_id`'s own docstring overstates its
    guarantee — it says "stable and content-addressed," but it's actually
    `f"{source_id}:{chunk_index}"` (positional within a content-addressed
    source, not a hash of the chunk's own text). Two different chunker
    configurations over the same file could produce the identical
    chunk_id sequence with different text at each index — harmless today
    (nothing relies on chunk_id detecting a text change at a fixed
    position), but `ragcore.ingestion.pipeline.group_fingerprint`
    (Phase 1) had to hash chunk *text*, not `chunk_id`, specifically
    because of this gap — see its module docstring and
    `docs/adr/0002-ingest-checkpointing-opt-in.md`. Worth a docstring
    correction next time that file is touched for an unrelated reason;
    not fixed here to stay within this pass's actual scope.

## 4. Mimisi / Vastu Shastra boundary check

Per project context: `ragcore` is meant to be a **domain-agnostic shared
base** for two unrelated downstream products — Mimisi (philosopher chat) and
Vastu Shastra (a rule engine). The codebase already states and applies its
own test for this explicitly, in `CLAUDE.md`'s conversation-layer section:
*"would this make sense applied to an unrelated corpus in a different domain
too?"*

**No violations found.** Specifically:
- `conversation/` ships `PersonaChatSession`, generic session/query-rewriting
  primitives — no "philosopher" concept anywhere in the type system.
- A prior version of the codebase *did* ship a domain-specific
  `DebateSession` directly in the shared package; it was identified as
  failing the boundary test and **removed** (`git log`: `c38078c Remove
  DebateSession from the shared base layer`) — this is direct evidence the
  boundary is being actively enforced, not just aspirational.
- `claims/` (claim extraction/agreement/conflict) is phrased generically
  ("source claims," "authors") — nothing Mimisi- or philosopher-specific,
  despite clearly being motivated by that product.
- Nothing in any module references a rule engine, Vastu Shastra, tenant, or
  product name.

The one latent risk (not a violation, a documented limitation — see §3.5) is
that persona/tenant isolation is an *assumption*, not an *enforced
invariant*, in the claims and conversation layers — something to keep front
of mind when the two products eventually share infrastructure.

## 5. Gap list — build order for Phase 1 onward

Ordered to match the spec's own phase sequence; items already fully done
are omitted.

**Phase 1 (ingestion): done, except item 5 (deliberately deferred).**
1. ✅ HTML loader (`loaders/html.py`).
2. ✅ OCR fallback for scanned PDFs (`PdfLoader(ocr=True)`, `ragcore[ocr]`).
3. ✅ Stage-level checkpointing inside `Corpus.ingest()`
   (`checkpoint_store=...`, opt-in) with `ingest_many` dead-letter handling.
4. ✅ Page-level content hash (`Block.page_hash` / `ChunkMeta.extra["page_hash"]`).
5. ⬜ (deferred) `tenant_id`/`edition`/`language`/`ingest_version` fields on
   `ChunkMeta` — deferred until Phase 7 design is settled, so these aren't
   bolted on twice.
6. ✅ A second `Chunker` strategy — `SemanticChunker`
   (`ragcore/chunking/semantic.py`).

**Phase 2 (embedding/index): items 7-10 done, 11 and 12 mostly done (12
with a stated testing-confidence caveat on the Postgres half) — Phase 2
is functionally complete, only 1M/10M benchmark runs and
Postgres-against-a-live-server verification remain as real, open work.**
7. ✅ Wrap every storage-layer backend exception in `ragcore.errors.StoreError`
   (`ragcore.errors.wrap_backend_errors`, `docs/adr/0004`).
8. ✅ Second embedding provider — `OpenAIEmbedder`
   (`docs/adr/0003-second-embedding-provider.md`).
9. ✅ Cost tracking — `ragcore.embedding.cost.UsageTracker`.
10. ✅ Embedding cache keyed by `(content_hash, model_name, dim)` —
    `ragcore.embedding.cache`; zero-downtime re-embedding migration falls
    out of the key shape (see the Phase 2 coverage section above).
11. 🟡 Index lifecycle: compaction/tuning hooks on `LanceDBStore` — ✅
    (`create_vector_index`/`compact`/`index_stats`); benchmark harness —
    ✅ tool (`scripts/benchmark_lancedb.py`), run up to 50k chunks
    (`docs/BENCHMARKS.md`); **1M/10M runs not yet executed** — real
    remaining work, needs dedicated time/hardware, not a missing
    mechanism.
12. 🟡 S3-compatible raw-file storage — ✅ (`ragcore.raw_files.S3RawFileStore`).
    Postgres metadata/job store — 🟡 (`PostgresIngestCheckpointStore`,
    **not yet run against a live server** — `docs/adr/0005`); extending
    the pattern to the other SQLite-backed stores remains open.

**Phase 3 (retrieval): done.**
13. ✅ Standalone query understanding layer — `ragcore.retrieval.query_expansion`.
14. ✅ Query router — `ragcore.retrieval.routing`.
15. ✅ Token-budget-aware context packer — `ragcore.generation.context_packer`.

**Phase 4 (GraphRAG):**
16. Unify/extend the entity-type taxonomy toward the book-tuned schema the
    spec names (characters/people/places/works/events/themes/claims) —
    today entities, concepts, and claims are three separate layers with
    separate schemas.
17. Gleaning passes (multi-round extraction) for `graph/`/`concepts/`.
18. Kuzu graph-store adapter behind the existing `GraphStore` protocol
    (today: SQLite only).
19. Leiden (vs. current Louvain) if hierarchical multi-level community
    detection becomes a real requirement — low priority, Louvain is working
    and tested.
20. Named search-mode selector (local/global/DRIFT) composing the existing
    `GraphRetriever`/`ConceptRetriever` layers explicitly, rather than
    always-on stacking.

**Phase 5 (generation/grounding):** 21. Model routing (cheap vs. strong per
call site) and prompt caching on the `anthropic` client calls. 22. Streaming
(SSE) — blocked on Phase 8's server existing at all.

**Phase 6 (eval/observability):** 23. Wire `eval_baseline.json` into an
actual CI gate (needs a CI pipeline first — none exists). 24. Synthetic
gold-question generation from a real book. 25. OpenTelemetry tracing +
structured logging — currently **zero log calls anywhere in `ragcore`**;
this is the single biggest observability gap and should land early, since
every later phase benefits from it retroactively.

**Phase 7 (multi-tenancy):** everything, from scratch — this is the next
major phase boundary, not an extension of existing code. Start with
`tenant_id` propagation through `ChunkMeta` and every store schema, since
every other Phase-7 item (quotas, isolation, metering) depends on that
field existing first.

**Phase 8 (API/SDK/deployment):** everything, from scratch, after Phase 7's
tenancy model is settled (an API surface built before tenant_id exists would
need rework).

**Phase 9 (docs):** ADR backfill for major decisions already made (this
pass seeds `docs/adr/0001-...`); Mermaid architecture diagram; cookbook
promoted from the existing `scripts/*_demo.py` files, which are already
most of the way there.
