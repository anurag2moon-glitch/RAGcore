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
of the target spec are substantially (not fully) covered already**,
Phase 7 is done for everything tractable as library code (tenant_id
foundation + enforcement, hard-delete, semantic caching, PII/
prompt-injection defenses, licensing flags — what's left needs either a
real server/auth context or is a deliberately deferred design question),
Phase 8 has a reference FastAPI server (`server/`) with its free OpenAPI
spec, a Python SDK, a TypeScript SDK, and an MCP server all done and
tested (gRPC, Docker/K8s/Redis/feature-flags/load-tests remain open —
the last four need infrastructure this sandbox can't validate, not
guessed-at, so they weren't attempted), and **Phase 9's entire original
gap list is done** — ADR backfill, a backup/restore runbook, a Mermaid
system diagram, a data-model reference, a configuration reference, and
a cookbook, every one of them checked directly against the actual code
and (where runnable) actually re-run before being written down. See §2
for the precise breakdown.

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
| `retrieval/` | Query → ranked `Retrieved[]`, composable decorator chain | `VectorRetriever`, `HybridRetriever` (RRF + BM25 via FTS5), `GraphRetriever`, `ConceptRetriever`, `HierarchyRetriever`, `IterativeRetriever` (multi-hop), `CrossEncoderReranker`, `MMRReranker`, `QueryExpansionRetriever` (HyDE/multi-query/decomposition), `RoutedRetriever` (per-query mode routing), `DriftRetriever`/`graph_search_modes()` (named local/global/DRIFT); `fusion.rrf_merge` shared by all the RRF-fusing layers; `stack.default_stack()` wires the always-on default chain | One test file per layer + `test_default_stack.py` + integration load/query tests |
| `lexical/` | BM25 full-text index | `Fts5LexicalStore` (stdlib SQLite FTS5, zero new deps; `tenant_id`-aware search, drop-and-rebuild migration since fts5 refuses `ALTER TABLE`) | `test_lexical_fts5.py`, `test_lexical_pipeline.py` |
| `graph/` | Entity/relation extraction + GraphRAG store | `ClaudeEntityExtractor` (book-tuned taxonomy, opt-in `gleaning_rounds`), `SqliteGraphStore` + `KuzuGraphStore` (WAL/locked and embedded-property-graph, both merge-not-replace), `EmbeddingEntityResolver`, shared `merge.merge_extra` | ~19+ tests incl. concurrency, cross-book resolution, a real P&P fixture; `test_kuzu_graph_store.py` (parametrized against both stores) |
| `concepts/` | Abstract concept/theme extraction + community detection + summaries | `ClaudeConceptExtractor` (opt-in `gleaning_rounds`), `SqliteConceptStore`, Louvain or Leiden via `networkx` (`concepts` extra), `EmbeddingConceptMapper`/`ClaudeConceptMapper` | `test_communities.py`, `test_communities_leiden.py`, `test_concept_*.py` (extraction/removal/resume/retrieval/store) |
| `contextual/` | Per-chunk situating blurb before re-embedding | `ClaudeContextWriter`, `SqliteContextStore` | `test_contextual_embeddings.py`, `test_contextual_resume.py` |
| `hierarchy/` | chunk→section→chapter→book summarization | `ClaudeHierarchySummarizer`, `SqliteHierarchyStore` | `test_hierarchy*.py` (5 files) |
| `claims/` | Atomic claim extraction + agree/conflict/extends linking (storage only, no retriever) | `ClaudeClaimExtractor`, `ClaudeClaimLinker`, `SqliteClaimStore` | `test_claims.py`, `test_claim_linking.py`, `test_claim_removal.py` |
| `conversation/` | Session transcript + history-aware query rewriting on top of `Corpus.ask()` | `InMemoryTranscriptStore`, `HistoryAwareQueryRewriter`, `PersonaChatSession` | `test_conversation_chat_session.py`, `test_conversation_transcript.py` |
| `generation/` | Context + query → `Answer` | `ClaudeGenerator`, `context_packer.pack_context` (opt-in token-budget packing, dedupe, neighbor expansion, document ordering) | `test_generation.py`, `test_context_packer.py`, `test_claude_generator_context_packing.py`; live test opt-in |
| `verification/` | Post-hoc per-claim faithfulness check | `ClaudeVerifier`, `VerifyingGenerator` (decorator) | `test_verification.py` |
| `eval/` | General-purpose retrieval/groundedness eval harness for a *consumer's* corpus | `run_eval()`, `GoldQuestion` JSONL format, recall/precision/nDCG/MRR/latency/tokens + optional groundedness, `ClaudeGoldSetGenerator` (synthetic QA from a real book) | `test_eval_*.py` (5 files); distinct from `tests/integration/eval/`, ragcore's own internal regression harness, now wired into `.github/workflows/ci.yml` on every push/PR |
| `versioning/` | Incremental re-ingest (diff/sync), snapshots, rollback | `IngestionManifest`, `plan_sync`/`sync_corpus`, `LanceDBStore.snapshot()`, SQLite online-backup | `test_ingestion_manifest.py`, `test_sync_corpus.py`, `test_snapshot_rollback.py` |
| `reranking/` | Cross-encoder rerank scoring | `VoyageReranker` (reuses Voyage API, zero new dep) | `test_voyage_reranker.py`, `test_cross_encoder.py` |
| `tenancy/` | Phase 7 multi-tenancy enforcement on top of `ChunkMeta.tenant_id` | `TenantScopedCorpus` (wraps one `Corpus` + fixed `tenant_id`, forces it on every call, raises `TenancyError` on conflict; vector-store path only — see `docs/adr/0006`), `hard_delete()` (per-tenant wipe across the vector store + optional auxiliary stores, safe for Tier 2 stores too — see `docs/adr/0007`) | `test_tenancy.py` |
| `privacy/` | Phase 7 PII detection — opt-in scan over an already-ingested `Corpus` | `PiiDetector` protocol, `RegexPiiDetector` (email/phone/SSN/credit-card, Luhn-checked, zero new dep), `scan_corpus()` | `test_privacy_regex_detector.py`, `test_privacy_pipeline.py` |
| `injection/` | Phase 7 prompt-injection defenses — ingestion-time audit + generation-time guard | `InjectionDetector` protocol, `RegexInjectionDetector` (5 common phrasings), `scan_corpus()`, `GuardedGenerator` (`Generator` decorator, `flag`/`exclude` modes, never rewrites `Chunk.text`) | `test_injection_regex_detector.py`, `test_injection_pipeline.py`, `test_injection_guard.py` |
| `caching/` | Phase 7 semantic/result cache | `ResultCache` protocol, `InMemoryResultCache` (TTL + tenant-tagged entries, zero new dep), `CachingRetriever` (`Retriever` decorator) | `test_caching_memory.py`, `test_caching_retriever.py` |
| `rules/` | Phase 10: domain-pluggable rule extraction + deterministic rule base — versioned schema, 3-pass extraction pipeline, pure-Python evaluator (+ JSON-Logic-style compiler), dedup/contradiction/exception consistency layer, review workflow, serving query API, pack export/import | `ClaudeRuleCandidateDetector`/`ClaudeRuleExtractor`/`ClaudeRuleVerifier`, `SqliteRuleStore` (immutable-version-on-edit, not merge-not-replace), `ReviewQueue`, `is_low_quality_ocr` | `test_rules_*.py` (12 files, 201 tests) |
| `domains/lal_kitab/` | Phase 10 §7/§6: first concrete `DomainPack` + its hand-labeled gold set (sibling package, like `server/`) | `LalKitabOntologyNormalizer` (real Hindi/Sanskrit transliteration tables), `LalKitabPack` (185-entry generated `fact_types()`) | Own suite, 208 tests, run from `domains/lal_kitab/`; own CI job (`test-lal-kitab`) |

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

### Phase 4 — GraphRAG — **complete as of 2026-10-01**
- ✅ Entity/relation extraction with typed schemas — `ClaudeEntityExtractor`'s
  prompt now names the book-tuned taxonomy (character/person/place/work/
  organization/event), with "theme"/"claim" explicitly excluded (handled by
  `concepts/`/`claims/` at their own granularity, by documented design —
  see CLAUDE.md, not an oversight). `entity_type` stays open vocabulary at
  the protocol level, unchanged. Gleaning passes — ✅
  (`gleaning_rounds=...`, opt-in, on both `ClaudeEntityExtractor` and
  `ClaudeConceptExtractor`).
- ✅ Entity resolution/canonicalization — unchanged (`EmbeddingEntityResolver`).
- ✅ Graph storage behind an interface — `KuzuGraphStore` now ships
  alongside `SqliteGraphStore`, proven interchangeable via a parametrized
  test suite run against both (Neo4j/Postgres adapters remain open).
- ✅ Hierarchical community detection — `detect_communities(algorithm=...)`
  now supports Leiden alongside the default Louvain (networkx's native
  `leiden_communities`, no new dependency); still one flat level, not
  explicitly multi-level (Leiden's main advantage over Louvain —
  guaranteed-well-connected communities — isn't yet exercised by anything
  multi-level in this codebase).
- ✅ Search modes: local/global/DRIFT — `ragcore.retrieval.drift`
  (`DriftRetriever`, `graph_search_modes()`), composing with Phase 3's
  `RoutedRetriever` for per-query mode selection. DRIFT is a stated
  simplification (fused-single-call local+global via RRF, not the GraphRAG
  paper's iterative primer-then-decompose-then-refine algorithm) — see its
  module docstring for the exact scope.
- ✅ Incremental graph updates, no full rebuild — unchanged (`sync_corpus` +
  `remove_chunk_ids`/`remove_source`, now also implemented on `KuzuGraphStore`).
- ✅ Provenance on every node/edge back to chunk — content-addressed ids +
  `source_chunk_ids` unioned on every merge

### Phase 5 — Generation and grounding — **done as of 2026-10-06, except item 22 (blocked)**
- 🟡 Inline citations at chunk level — ✅ (unchanged); **page/chunk clickable
  spans** — still a frontend concern outside a library's scope, unchanged.
- ✅ Faithfulness verification — unchanged (`ClaudeVerifier`/`VerifyingGenerator`).
- 🟡 Streaming (SSE) — ⬜, still blocked on Phase 8's server existing at
  all (not attempted — correctly sequenced, not forgotten). Prompt caching
  — ✅ (`ragcore.prompt_caching`, opt-in `enable_prompt_caching=...` on
  `ClaudeGenerator`/`ClaudeEntityExtractor`/`ClaudeConceptExtractor` — the
  three components with genuine, non-speculative repeat-content reuse;
  the other ~10 Claude-backed components don't have it yet, stated as
  open follow-up, not claimed done). Model routing — ✅
  (`ragcore.model_routing.DEFAULT_ROUTING`: 10 components now default to
  a cheap tier, 3 final-answer-facing ones stay on the strong tier
  unchanged). Fallbacks — 🟡 unchanged (retry-with-backoff only, no
  cross-provider fallback — not attempted this pass).
- 🟡 Abstention when evidence is insufficient — the default system prompt
  instructs the model to say so explicitly; no mechanical enforcement (it's
  prompt-level, not a checked behavior)

### Phase 6 — Evaluation and observability — **numbered gap items done as of 2026-10-06; two sub-items remain open**
- ✅ Eval harness — recall@k/MRR/nDCG/precision/groundedness (unchanged)
  **plus**, added this pass: synthetic QA generation from a real book
  (`ragcore.eval.gold_generation.ClaudeGoldSetGenerator`) and
  latency/cost per query (`RetrievalMetrics.latency_p50/p95/p99_ms`,
  `total_tokens_used`; `QuestionResult.latency_ms`). Answer-relevance
  metric (distinct from groundedness) — ⬜, still open, not attempted.
- ✅ CI regression gate — `.github/workflows/ci.yml` runs the full suite
  (including the already-existing `test_eval_quality.py` baseline
  comparison) on every push/PR; that workflow failing *is* the gate.
  Caught a real bug while validating it against a clean install:
  selectolax 1.0 broke `EpubLoader`/`HtmlLoader`, fixed (see commit
  `ec83168`). Lint runs report-only (141 pre-existing findings, not this
  pass's to clean up en masse).
- ⬜ A/B framework for chunking/embedding/reranker/prompt variants — still
  doesn't exist as a dedicated harness; `ragcore.eval.report.compare()`
  diffs two already-computed `EvalReport`s, which covers part of this,
  but there's no variant-running convenience on top of it yet.
- 🟡 Structured logging — ✅ (`logging.getLogger("ragcore.<module>")`,
  no handler configured by the library; wired into `Corpus`,
  `build_graph`, `build_concepts`, `detect_communities`/
  `build_community_summaries`, both embedding providers' retry loops).
  OpenTelemetry tracing, per-tenant dashboards, Langfuse-style trace
  viewing — ⬜, still open; the other ~10 Claude-backed components
  (extractors' own call-level detail, router, judge, rewriter, verifier,
  generator) don't have logging yet either.

### Phase 7 — Multi-tenancy and platform — 🟡 every library-tractable item done (2026-10-07); auth/RBAC/quotas and encryption/audit-logs need a real server (Phase 8), Tier 2 tenant_id is a deliberately open design question
- ✅ `tenant_id` foundation — `ChunkMeta.tenant_id: str | None = None`
  (opt-in, additive), wired through `Corpus.ingest()`/`ingest_many()`'s
  new `tenant_id=...` param, and through `LanceDBStore`'s schema +
  `_FILTERABLE_FIELDS` (so `filters={"tenant_id": "..."}` works against
  the primary vector store; an old on-disk table without the column
  migrates transparently on open). See
  `docs/adr/0006-tenant-id-foundation-not-enforcement.md` for exactly
  what this does and doesn't cover.
- ✅ **Isolation enforcement for the vector-store path** —
  `ragcore.tenancy.TenantScopedCorpus` wraps a `Corpus` + fixed
  `tenant_id` and forces it on every `ingest`/`ingest_many`/`retrieve`/
  `ask`/`get`/`sources`/`remove`/`count` call, raising `TenancyError` on
  any caller-supplied value that disagrees rather than silently
  overriding or silently trusting it.
- ✅ **`HierarchyRetriever` fixed to actually respect the filter** —
  found while finishing item 28a: adding `tenant_id` to
  `SqliteHierarchyStore` alone did **not** close a leak.
  `HierarchyRetriever.retrieve()` called `nodes_by_level()` (every
  tenant's nodes) and never re-checked the result against `filters` at
  all — unlike `HybridRetriever`'s lexical-only hits, which already got
  a Python post-filter, nothing downstream of `HierarchyRetriever`
  re-validates a filter against its synthetic node results either, so
  this was a genuine cross-tenant leak, not a recall-at-scale concern
  like the lexical case. Fixed: `nodes_by_level()`'s candidates are now
  filtered by `tenant_id` (via the new shared
  `ragcore.retrieval._filters.tenant_id_from_filter`) before scoring,
  and `_node_chunk()` now stamps the synthetic chunk's own
  `tenant_id` from `node.tenant_id` (it never did before). Proven with a
  test that reproduces the exact leak shape and confirms it's closed.
- 🟡 `GraphRetriever`/`ConceptRetriever` — **not yet checked or fixed**.
  Their own stores (`SqliteGraphStore`/`KuzuGraphStore`,
  `SqliteConceptStore`) are Tier 2 (item 28b, below) and don't carry
  `tenant_id` at all yet, so the same category of fix isn't possible
  for them until that lands — stacking either onto a wrapped `Corpus`
  is not protected by `TenantScopedCorpus` today.
- 🟡 `tenant_id` on every *other* store — splits into two genuinely
  different problems, not six repeats of the same bolt-on (see
  `docs/adr/0007-tenant-id-on-remaining-stores-two-tiers.md`):
  - **Tier 1 (passive column is correct)**: ✅ all four done —
    `Fts5LexicalStore` (tenant_id column + `HybridRetriever` SQL-level
    pushdown via `_tenant_id_from_filter`; migration isn't LanceDB's
    `add_columns()` pattern — FTS5 virtual tables refuse `ALTER TABLE`
    outright, confirmed directly, so this one drops-and-rebuilds
    instead), `SqliteContextStore`, `SqliteHierarchyStore`,
    `SqliteClaimStore` (the latter three are ordinary SQLite tables:
    `PRAGMA table_info()` + `ALTER TABLE ADD COLUMN`, no drop-and-rebuild
    needed).
  - **Tier 2 (passive column would be actively unsafe)**:
    `SqliteGraphStore`/`KuzuGraphStore`, `SqliteConceptStore` — entity/
    relation/concept ids are content-addressed over bare normalized
    text with no source scoping *by design* (so "Bingley"/"Mr. Bingley"
    merge across chapters); two tenants' same-named entity or concept
    compute the same id and merge-not-replace write logic would merge
    their provenance into one shared row today, a real cross-tenant
    leak a passive filter column cannot fix. Needs `tenant_id` baked
    into the id itself or a compound key, deliberately left as an open
    design question rather than shipped half-right.
  - Only the vector store has `tenant_id` today, and only the
    vector-retrieval path is enforced by `TenantScopedCorpus` — do
    **not** wire `GraphRetriever`/`ConceptRetriever` onto a
    multi-tenant `TenantScopedCorpus` until Tier 2 is resolved.
- 🟡 API keys — ✅ `server/app/auth.py` (`ApiKeyStore` protocol,
  `InMemoryApiKeyStore` reference implementation, `RAGCORE_API_KEYS`-
  configured, opt-in and backward compatible — see
  [ADR 0012](adr/0012-api-key-auth-and-multi-tenant-routing.md)). This
  is what unlocked real multi-tenant HTTP routing on `server/` (next
  bullet). Rate limiting — ✅ `server/app/rate_limit.py` (token bucket
  per tenant_id/client IP, `RAGCORE_RATE_LIMIT_PER_MINUTE`-configured,
  same opt-in shape). Usage metering — ✅ `server/app/usage.py`
  (`GET /usage`, request counts per tenant — deliberately scoped to
  counts, not token/cost, since this server shares one embedder across
  every tenant; see that module's own docstring). Per-tenant quotas —
  ✅ `server/app/quota.py` (`RAGCORE_TENANT_QUOTAS`, lifetime-total
  caps, `402` when exceeded, reuses the usage meter's own counts).
  JWT/OAuth, RBAC — ⬜, still open; everything else in item 30 is done.
- ✅ Semantic/result cache with tenant-scoped invalidation —
  `ragcore.caching` (`ResultCache` protocol, `InMemoryResultCache`
  default, `CachingRetriever` decorator). `invalidate(tenant_id=...)` is
  a real targeted operation, not just TTL expiry or clear-everything.
  Nothing auto-invalidates on a corpus mutation — a consumer calls
  `invalidate()` itself after whatever change it cares about.
- 🟡 Data lifecycle — ✅ hard delete:
  `TenantScopedCorpus.hard_delete()` wipes one tenant's chunks from the
  vector store plus every row derived from them in whichever optional
  auxiliary stores (graph/concept/claim/context/hierarchy/lexical) a
  caller passes, reusing each store's own existing `remove_source()`
  unchanged — safe even for the Tier 2 stores, proven by a test
  simulating two tenants sharing one content-addressed entity id (see
  commit 38e1b01). Export and retention policy — ⬜, not attempted.
- 🟡 PII detection — ✅ `ragcore.privacy` (`PiiDetector` protocol,
  `RegexPiiDetector` default: email/phone/SSN/credit-card, Luhn-checked;
  `scan_corpus()`, a standalone pass over an already-ingested `Corpus`,
  same shape as `run_eval()`/every `build_*()` pass).
- 🟡 Prompt-injection defenses — ✅ `ragcore.injection` (`InjectionDetector`
  protocol, `RegexInjectionDetector` default: 5 common phrasings —
  ignore/disregard-instructions, role-override, reveal-system-prompt,
  end-of-context markers; `scan_corpus()` for ingestion-time audit;
  `GuardedGenerator`, a `Generator` decorator screening retrieved
  context before generation — `mode="flag"` stamps
  `extra["injection_flagged"]` without touching `Chunk.text`,
  `mode="exclude"` drops a flagged chunk from context entirely rather
  than editing it, since silently rewriting it would leave the
  Generator-visible text and a `Corpus.get(chunk_id)` citation lookup
  inconsistent).
- ✅ Per-document licensing flag — `ChunkMeta.license_id` (named to
  avoid shadowing the `license` builtin — confirmed via ruff's A002,
  not guessed), same opt-in/additive shape as `tenant_id`:
  `Corpus.ingest(..., license_id=...)`, filterable via `LanceDBStore`.
  Mechanism only, same "not an enforced guarantee" positioning as
  `tenant_id` — ragcore filters on it if asked, it never refuses to
  generate from a flagged chunk on its own.
- ⬜ Encryption at rest/in transit config, audit logs — none of this
  exists yet; both need an actual request/server context (Phase 8)
  this library doesn't have.

### Phase 8 — API, SDKs, deployment — 🟡 FastAPI service started (2026-10-07), everything else still open
- ✅ **Reference FastAPI server** — `server/` (sibling package to
  `ragcore/`, own `pyproject.toml`, zero new dependencies added to
  `ragcore`'s own — see `docs/adr/0008-api-server-as-sibling-package.md`).
  `GET /health`, `POST /ingest` (real multipart upload), `POST /retrieve`,
  `POST /ask`, `GET /sources`, `DELETE /sources/{source_id}`.
  Single-tenant and unauthenticated by default; **real multi-tenant
  auth now exists** — set `RAGCORE_API_KEYS` and every route is scoped
  through `TenantScopedCorpus` (`401`/`403` wired in) — see
  `docs/adr/0012-api-key-auth-and-multi-tenant-routing.md`.
- ✅ **OpenAPI spec** — free once the server above exists; FastAPI
  generates it automatically from the route/model definitions
  (`server/app/main.py`/`server/app/models.py`), visit `/docs`. No
  separate authoring pass was needed — the gap list named this as if it
  were independent work, it isn't.
- ✅ **Python SDK** — `sdk/python/` (`ragcore_client.RagcoreClient`, a
  thin typed `httpx` wrapper over the server's endpoints; constructor-
  injectable `http_client`, same discipline as every other swappable
  component here). Tested against the real `server/` app in-process via
  FastAPI's `TestClient`.
- ✅ **MCP server** — `mcp-server/` (`mcp_server.server.build_server`,
  four tools: `ingest_file`/`retrieve`/`ask`/`list_sources`, via the
  official `mcp` SDK's `MCPServer`). Deliberately duplicates `server/`'s
  single-tenant corpus-provider shape rather than importing it — two
  independently launched artifacts (an MCP client vs. an HTTP caller)
  shouldn't be coupled through one importing the other.
- ✅ **TypeScript SDK** — `sdk/typescript/` (`RagcoreClient`, zero
  runtime dependencies — Node's built-in `fetch`/`FormData`/`Blob`).
  Tested against a *real* `server/` process over a real socket
  (`server/tests/run_test_server.py`) — no in-process ASGI bridge is
  reachable from a Node test runner the way FastAPI's `TestClient`
  gives the Python SDK one.
- ⬜ gRPC, Docker/Helm/k8s manifests, Redis, feature flags, load tests
  (Locust/k6) — none of this exists yet.
- ✅ **Backup/restore runbook** — `docs/RUNBOOK.md`, researched directly
  against the code and its test suite (exact file/line sourced for
  every claim). Documents the real `create_snapshot()`/`rollback()`
  orchestration, LanceDB's actual non-destructive tag/restore
  semantics, and five real gaps the library doesn't automate (the
  `IngestionManifest` itself has no restore path; ingest checkpoint
  stores are excluded entirely; the cross-store sequence isn't atomic;
  `rollback()` must be called with the same stores `create_snapshot()`
  had or a store is silently skipped; retention only prunes LanceDB
  tags).

### Phase 9 — Docs — ✅ original gap list complete (2026-10-07)
- ✅ ADR backfill (3 of likely several more worth adding later) —
  `docs/adr/0009-library-not-application.md`,
  `0010-content-addressed-identity.md`, `0011-graph-merge-not-replace.md`,
  each explicitly marked "backfilled" with today's date rather than
  claiming retroactive authorship.
- ✅ Runbook — `docs/RUNBOOK.md` (backup/restore; see Phase 8 item 38
  above for what it covers).
- ✅ Mermaid system diagram — `docs/DIAGRAM.md`, two flowcharts,
  rendered and visually verified via `@mermaid-js/mermaid-cli`.
- ✅ Data-model reference — `docs/DATA_MODEL.md`, every dataclass a
  consumer is likely to construct/read/persist.
- ✅ Configuration reference — `docs/CONFIGURATION.md`, confirmed
  directly that `ragcore` itself reads zero environment variables.
- ✅ Cookbook — `docs/COOKBOOK.md`, promoted from `scripts/*_demo.py`;
  every script re-run against this exact commit before being written
  up, not described from memory.
- `CLAUDE.md` + `README.md` remain the unusually thorough *decision
  logs* they always were; the ADR directory now has 11 entries on top
  of them. Nothing in this phase's original gap list is open — further
  documentation work (more ADR backfill, per-downstream-project
  cookbook entries once Mimisi/Vastu Shastra integration details
  exist) is tracked as open-ended improvement, not a numbered gap.

### Phase 10 — Rule extraction and rule base (domain-pluggable) — ✅ all 8 sections done (2026-10-07)

A new, 10th phase beyond the original 9-phase target spec, added mid-session
per a separate spec: extract hard, machine-evaluable rules from books (first
domain: Lal Kitab astrology), store them with provenance, serve them to a
deterministic engine. See `docs/adr/0013-rule-extraction-domain-pluggable-architecture.md`.

- ✅ **§1 Rule schema** — `ragcore/rules/types.py`. Pydantic + JSON Schema,
  versioned (`SCHEMA_VERSION`). `Condition` is a discriminated union
  (`FactPredicate` leaf / `ConditionGroup` AND-OR-NOT composite), compiling
  to a JSON-Logic-style DSL (`to_json_logic()`) verified, not assumed,
  semantically equivalent to `ragcore.rules.evaluator.evaluate_condition`
  via a 5000-case randomized cross-check against a real JSON Logic engine
  (`json-logic-qubit`, a dev-only dependency) — this caught two real
  compilation bugs (truthiness-vs-presence for `exists`, missing-fact
  null-propagation for every comparison operator) before they shipped.
  `rule_content_hash` is content-addressed over
  `domain|book_id|normalized_span|conditions|effect` — deliberately
  excluding chapter/page/chunk_ids/remedy/confidence/rule_type, so the
  same rule restated elsewhere collapses to one id, and a human *edit*
  (§4) deliberately does **not** reuse this hash (see §4 below).
- ✅ **§2 Extraction pipeline** — `ragcore/rules/claude.py` + `pipeline.py`.
  3 passes (`ClaudeRuleCandidateDetector`/`Extractor`/`Verifier`), chapter-
  grouped and checkpointed exactly like `build_concepts`/`build_claims`.
  An unresolvable fact/value (ontology can't normalize it) drops the whole
  candidate rule rather than storing a weaker one than the span states. A
  rule failing pass-3 verification is never written to the store at all —
  "reject anything unsupported" applied literally, upstream of and
  distinct from human review status. `ocr_quality.is_low_quality_ocr`
  (replacement-char ratio + alphabetic ratio heuristic — the honest option
  without `ragcore.loaders` carrying per-word OCR confidence through to
  `ChunkMeta`) excludes flagged chunks from extraction input; a chapter
  entirely flagged is skipped with no checkpoint, same "pick up for free
  later" discipline as a chapter with no concepts yet. 🟡 incremental
  re-extraction of changed chunks is covered by the existing fingerprint/
  checkpoint mechanism (unchanged chunks are a checkpoint hit), but
  nothing yet surgically re-extracts *only* a changed subset within an
  otherwise-unchanged chapter — out of scope until a real multi-run corpus
  surfaces whether that granularity is worth the complexity.
- ✅ **§3 Consistency layer** — `ragcore/rules/consistency.py`. Exact-match
  dedup (`find_duplicates`), contradiction detection (`find_contradictions`
  — same flat condition set, same effect category, opposite polarity), and
  exception detection (`find_exceptions` — an EXCEPTION/CANCELLATION rule
  whose flat condition set is a proper superset of another's). Stated
  plainly, not hidden: both contradiction/exception detection require a
  *flat conjunction* of facts (general AND/OR/NOT subsumption is
  SAT-equivalent, out of scope) — an `or`/`not` condition is skipped, never
  mishandled. Conflicts persist via `RuleStore.put_conflicts`/
  `conflicts_for_rule`, insert-or-ignore by content-addressed
  `conflict_id` — "never silently overwrite."
- ✅ **§4 Review workflow** — `ragcore/rules/review.py`. `ReviewQueue.list_pending`
  (lowest-confidence-first, filterable by domain/book/confidence),
  `approve`/`reject` (an explicit, narrow in-place mutation of
  `review_status` only — it's workflow metadata, never part of
  `rule_content_hash`'s inputs, so this doesn't violate the
  immutable-version-on-edit discipline rule *content* gets), `edit`
  (genuinely creates a new version via `RuleStore.supersede`, with a new
  `rule_id` hashed over the *entire* edited content plus
  `supersedes=old.rule_id` — deliberately not `rule_content_hash`, which
  is too coarse for an edit: it ignores fields like `confidence`/`remedy`
  that a reviewer might be the one changing). Every action appends one
  `ReviewAuditEntry` in the same transaction as the mutation it describes.
- ✅ **§5 Rule serving** — `ragcore/rules/evaluator.py` (pure, no LLM;
  `evaluate_condition`/`match_rules`/`validate_facts`) + `serving.py`
  (`match_with_provenance` — matches a fact set, groups results by
  already-stored conflicts, surfaces which matched rules are overridden,
  verified-only-by-default per §4) + `packs.py` (`export_pack`/`import_pack`
  as versioned JSON bundles; a tenant-private bundle refuses import under a
  *different* tenant_id outright, rather than risk a silent cross-tenant
  `rule_id` collision — `rule_content_hash` doesn't include `tenant_id` by
  design). 🟡 hybrid mode (RAG retrieval for explanation text + rule
  matching for facts, citations to book/page) is not yet built as its own
  function — deferred to land alongside a cookbook example, since it's a
  thin composition over `Corpus.ask()` + `match_with_provenance` rather
  than new mechanism.
- ✅ **§7 Domain pluggability** — `ragcore/rules/domain.py` (`DomainPack`/
  `OntologyNormalizer`/`FactTypeSpec` protocols) + the first concrete pack,
  `domains/lal_kitab/` (own sibling package, ADR 0008's pattern). Real
  Hindi/Sanskrit transliteration tables (not placeholder aliases),
  `fact_types()` generated (185 specs) rather than hand-enumerated.
  Proven end to end against the fully generic `ragcore.rules` layer with
  zero core changes (`domains/lal_kitab/tests/test_integration.py`).
- ✅ **§6 Evaluation** — `ragcore/rules/eval.py` (domain-agnostic metrics:
  `evaluate_extraction`/`evaluate_verification`/`evaluate_conflict_detection`/
  `evaluate_rules`) + `domains/lal_kitab/tests/gold.py` (a hand-authored,
  explicitly-synthetic 3-chapter gold set — not excerpted from any real
  published book, the same "synthetic fixture" discipline
  `tests/integration/`'s own 8 synthetic books already follow) and
  `test_eval_gold.py`, the CI regression gate (new `test-lal-kitab` job in
  `.github/workflows/ci.yml`). Deliberately not a live-quality benchmark
  against a real Claude run — a fixed, deterministic fixture, so a
  failure here means a regression in the ontology-to-schema construction
  path or in the eval scoring logic itself, not LLM variance. One real
  bug this caught before being trusted: a missed gold rule was scored
  against an unrelated sibling rule sharing its chunk, not treated as
  "nothing extracted" — fixed in `_candidate_match`.
- ✅ **§8 Integration and docs** — REST API (`server/app/rules_routes.py`:
  `list_rules`/`match_rules`/`approve`/`reject`/`edit`/`extract`), the
  same four operations as MCP tools (`mcp-server/`), and as methods on
  both SDKs (`sdk/python/`, `sdk/typescript/`). Tenant isolation for
  private packs: storage-level guard in `packs.py` (refuses cross-
  tenant pack import) **and** route-level filtering (every rule route
  shows global rules + only the caller's own tenant's private ones —
  verified against two real tenant API keys, not assumed). Extraction-
  job token-cost quota — `RAGCORE_EXTRACTION_TOKEN_QUOTAS`
  (`server/app/extraction_quota.py`), metered from the real
  `input_tokens + output_tokens` on every Claude response via
  `ClaudeRuleCandidateDetector`/`Extractor`/`Verifier`'s own
  `UsageTracker`, aggregated into `RuleExtractionReport.total_tokens_used`
  — checked before a job runs, recorded after. Schema reference
  (`docs/DATA_MODEL.md`), ontology reference (`docs/RULES_ONTOLOGY.md`),
  cookbook (`docs/COOKBOOK.md` + `domains/lal_kitab/examples/demo.py`).
  One real, pre-existing gap surfaced while wiring `/rules/extract`
  through `TenantScopedCorpus` — see "What's fragile" below.

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
11. **New from Phase 10**: every `build_*()` pipeline in `ragcore`
    (`build_graph`/`build_concepts`/`build_claims`/
    `build_contextual_embeddings`/`build_rules`) reads chunks via
    `corpus.store.get_by_source()` directly, bypassing
    `TenantScopedCorpus` entirely — none of them go through its
    wrapped methods, so in multi-tenant mode every one of them reads
    across *every* tenant's ingested chunks, not just one tenant's.
    Surfaced while wiring `/rules/extract` through a real
    `TenantScopedCorpus` in `server/app/rules_routes.py` (a genuine
    `AttributeError: 'TenantScopedCorpus' object has no attribute
    'store'` the first time any pipeline builder was actually called
    with one) — fixed *for that one route* by unwrapping back to the
    raw `Corpus` before calling `build_rules()`, with the limitation
    stated plainly in `server/README.md`, not retrofitted into
    `ragcore.rules.pipeline` itself (which would make it behave
    differently from every sibling pipeline builder, not actually
    close the systemic gap). The same `AttributeError` would reproduce
    identically for `build_graph`/`build_concepts`/`build_claims`/
    `build_contextual_embeddings` the moment anything calls one of
    them with a `TenantScopedCorpus` — this item exists so that's not
    rediscovered independently later. A real fix needs either a
    `.store` passthrough on `TenantScopedCorpus` (and a decision about
    whether that passthrough itself needs tenant-filtering, which
    `ragcore.tenancy.TenantScopedCorpus`'s own docstring already flags
    as an open question for other store/retriever paths) or a
    `get_by_source`-shaped method each pipeline builder calls through
    `Corpus`'s own interface instead of reaching into `.store`.

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

**Phase 4 (GraphRAG): done.**
16. ✅ Book-tuned entity-type taxonomy in `ClaudeEntityExtractor`'s prompt
    (themes/claims deliberately stay in `concepts/`/`claims/`, not unified —
    see Phase 4 coverage section above for why).
17. ✅ Gleaning passes — `gleaning_rounds=...` on both `ClaudeEntityExtractor`
    and `ClaudeConceptExtractor`.
18. ✅ Kuzu graph-store adapter — `ragcore.graph.KuzuGraphStore`.
19. ✅ Leiden — `detect_communities(algorithm="leiden")`.
20. ✅ Named search-mode selector — `ragcore.retrieval.drift`.

**Phase 5 (generation/grounding): item 21 done, item 22 blocked (correctly, not forgotten).**
21. ✅ Model routing (`ragcore.model_routing`) and prompt caching
    (`ragcore.prompt_caching`, on `ClaudeGenerator`/the two gleaning-capable
    extractors — not yet every Claude-backed component).
22. ⬜ Streaming (SSE) — blocked on Phase 8's server existing at all.

**Phase 6 (eval/observability): items 23/24 done, 25 half done.**
23. ✅ CI gate — `.github/workflows/ci.yml`.
24. ✅ Synthetic gold-question generation — `ClaudeGoldSetGenerator`.
25. 🟡 Structured logging — ✅ (stdlib `logging`, high-value paths).
    OpenTelemetry tracing — ⬜, not attempted.
26. ⬜ (unnumbered, from the coverage narrative) A/B framework for
    chunking/embedding/reranker/prompt variants, and an answer-relevance
    metric distinct from groundedness — both still open.

**Phase 7 (multi-tenancy): foundation steps 27/28a/29 done, everything
else (28b, 30+) still from scratch.**
27. ✅ `tenant_id` on `ChunkMeta` + `Corpus.ingest()`/`ingest_many()` +
    `LanceDBStore` (schema, filterable fields, transparent migration of
    an existing on-disk table). See
    `docs/adr/0006-tenant-id-foundation-not-enforcement.md`.
28. 🟡 `tenant_id` on every other store — two tiers, not one task; see
    `docs/adr/0007-tenant-id-on-remaining-stores-two-tiers.md`.
    28a. ✅ Tier 1, passive column is correct — all four done:
    `Fts5LexicalStore` (+ `HybridRetriever` SQL-level pushdown, commit
    3f1fd9b — migrated via drop-and-rebuild, since fts5 virtual tables
    refuse `ALTER TABLE`), `SqliteContextStore` (commit 8987bf9),
    `SqliteHierarchyStore` (commit 0bdedad), `SqliteClaimStore` (commit
    b76a7b8) — the latter three are ordinary SQLite tables, migrated via
    the simpler `PRAGMA table_info()` + `ALTER TABLE ADD COLUMN`.
    28b. ⬜ Tier 2, passive column would be actively unsafe (content-
    addressed ids with no source scoping, by design): graph store,
    concept store — needs `tenant_id` baked into the id or a compound
    key, an open design question, not a mechanical bolt-on.
29. ✅ Isolation enforcement for the vector-store path —
    `ragcore.tenancy.TenantScopedCorpus`, plus (found and fixed while
    closing out 28a) a real leak in `HierarchyRetriever`, which never
    re-checked its synthetic node results against `filters` at all —
    unlike the lexical channel's existing post-filter, this had no
    safety net anywhere downstream. Now fixed via a shared
    `tenant_id_from_filter` helper. `GraphRetriever`/`ConceptRetriever`
    are **not yet checked or fixed** — both read Tier 2 stores (28b)
    that don't carry `tenant_id` at all, so wiring either onto a
    multi-tenant `TenantScopedCorpus` today risks real cross-tenant
    leakage, not just a missed filter.
30. 🟡 API keys — ✅ `server/app/auth.py` (commit 707cd52). Rate
    limiting — ✅ `server/app/rate_limit.py` (commit f219ba6). Usage
    metering — ✅ `server/app/usage.py` (commit 62859c2), request
    counts only, not token/cost. Per-tenant quotas — ✅
    `server/app/quota.py` (commit cf78bb5), lifetime-total, not
    period-based. JWT/OAuth, RBAC — ⬜, still open.
31. ✅ Semantic/result cache with tenant-scoped invalidation —
    `ragcore.caching` (commit 9ac827f).
32. 🟡 Data lifecycle — ✅ hard delete:
    `TenantScopedCorpus.hard_delete()` (commit 38e1b01), proven safe
    against the exact Tier 2 shared-id danger item 28b names. Per-tenant
    export, retention policy — ⬜, still open.
33. 🟡 PII detection — ✅ `ragcore.privacy` (commit c85387a).
    Prompt-injection defenses — ✅ `ragcore.injection` (commit f7a7c68).
    Per-document licensing flag — ✅ `ChunkMeta.license_id` (commit
    709b9d4). Encryption at rest/in transit, audit logs — ⬜, still
    open; both need a real request/server context (Phase 8).

**Phase 8 (API/SDK/deployment): items 34/35 done, everything else (36+)
still from scratch.**
34. ✅ Reference FastAPI server (`server/`) + its free OpenAPI spec —
    commit 41b0bec. See `docs/adr/0008-api-server-as-sibling-package.md`.
35. 🟡 Python SDK (`sdk/python/`, commit faaeccd) — ✅. MCP server
    (`mcp-server/`, commit b0e85c2) — ✅. TypeScript SDK
    (`sdk/typescript/`, commit 2ba96e6) — ✅. gRPC — ⬜, still open.
36. ⬜ Docker Compose/Helm/K8s manifests, Redis, feature flags.
37. ⬜ Load tests (Locust/k6).
38. ✅ Backup/restore runbook — `docs/RUNBOOK.md`, commit eff4b1c.
39. ✅ Multi-tenant HTTP routing on top of `server/` — commit 707cd52,
    `docs/adr/0012-api-key-auth-and-multi-tenant-routing.md`. Every
    route scoped via `TenantScopedCorpus` once `RAGCORE_API_KEYS` is
    set; `401`/`403` wired in. JWT/OAuth/RBAC/quotas/rate-limiting
    (the rest of item 30) remain open.

**Phase 9 (docs): items 40/41 in progress, everything else (42+) open.**
40. 🟡 ADR backfill for major decisions already made — 3 done
    (commit a9aebaf: `0009` library-not-application, `0010`
    content-addressed identity, `0011` graph merge-not-replace), likely
    several more worth adding later (RRF fusion, the synthetic-chunk
    convention, the SQLite WAL+lock pattern are candidates — not
    attempted yet).
41. ✅ Runbook — `docs/RUNBOOK.md` (same as Phase 8 item 38 above; one
    piece of work, tracked under both since it closes a gap named in
    each phase's own list).
42. ✅ Mermaid architecture diagram — `docs/DIAGRAM.md`, commit
    d84cdc1. Both diagrams rendered and visually verified via
    `@mermaid-js/mermaid-cli`, not just syntax-checked.
43. ✅ Formal data-model reference — `docs/DATA_MODEL.md`, commit
    d84cdc1.
44. ✅ Configuration reference — `docs/CONFIGURATION.md`, commit e0dea16.
45. ✅ Cookbook — `docs/COOKBOOK.md`, commit 1638ff6, promoted from
    `scripts/*_demo.py` (every script re-run before being written up).

**All 45 numbered gap items from this audit's original Phase 0-9
breakdown are now done or explicitly, honestly marked as deferred to a
named follow-on condition** (Tier 2 tenant_id's open design question,
item 28b; auth-gated items 30/39; infrastructure this sandbox can't
validate, items 36/37). See §1 and each phase's own section above for
the precise status of what's done vs. deferred vs. genuinely still
open.
