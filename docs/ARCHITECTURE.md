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

Total: ~11,700 lines across `ragcore/`, 24 submodules, ~80 test files
(76 pass by default in ~7.7s; 4 more real-book tests opt-in via
`pytest -m gutenberg`; 4 live-API tests opt-in via `pytest -m live`).
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
| `loaders/` | File → `Document` | `pdf.py` (PyMuPDF, text only), `epub.py`, `docx.py`, `text.py` (txt/md) | `test_loaders_structure.py` — yes, incl. malformed-input handling |
| `chunking/` | `Document` → `ChunkGroup`s | `StructuralChunker` (chapter-aware, token-budget packing, overlap, paragraph/sentence/char-level oversized splitting) | `test_chunk_id_stability.py` + integration suite |
| `embedding/` | Text → vectors | `VoyageContextEmbedder` (voyage-context-4, contextualized, batched by inputs/chunks/tokens, retried) | `test_voyage_batching.py`; live test opt-in |
| `stores/` | Vector persistence + similarity query + filter DSL | `LanceDBStore` (embedded, on-disk, filter allowlist, snapshot/restore via native tags) | `test_store_swap.py`, `test_lancedb_snapshot.py`, integration concurrency/scale suite |
| `retrieval/` | Query → ranked `Retrieved[]`, composable decorator chain | `VectorRetriever`, `HybridRetriever` (RRF + BM25 via FTS5), `GraphRetriever`, `ConceptRetriever`, `HierarchyRetriever`, `IterativeRetriever` (multi-hop), `CrossEncoderReranker`, `MMRReranker`; `stack.default_stack()` wires the lot | One test file per layer + `test_default_stack.py` + integration load/query tests |
| `lexical/` | BM25 full-text index | `Fts5LexicalStore` (stdlib SQLite FTS5, zero new deps) | `test_lexical_fts5.py`, `test_lexical_pipeline.py` |
| `graph/` | Entity/relation extraction + GraphRAG store | `ClaudeEntityExtractor`, `SqliteGraphStore` (WAL, locked, merge-not-replace), `EmbeddingEntityResolver` | ~19 tests incl. concurrency, cross-book resolution, a real P&P fixture |
| `concepts/` | Abstract concept/theme extraction + community detection + summaries | `ClaudeConceptExtractor`, `SqliteConceptStore`, Louvain via `networkx` (`concepts` extra), `EmbeddingConceptMapper`/`ClaudeConceptMapper` | `test_communities.py`, `test_concept_*.py` (extraction/removal/resume/retrieval/store) |
| `contextual/` | Per-chunk situating blurb before re-embedding | `ClaudeContextWriter`, `SqliteContextStore` | `test_contextual_embeddings.py`, `test_contextual_resume.py` |
| `hierarchy/` | chunk→section→chapter→book summarization | `ClaudeHierarchySummarizer`, `SqliteHierarchyStore` | `test_hierarchy*.py` (5 files) |
| `claims/` | Atomic claim extraction + agree/conflict/extends linking (storage only, no retriever) | `ClaudeClaimExtractor`, `ClaudeClaimLinker`, `SqliteClaimStore` | `test_claims.py`, `test_claim_linking.py`, `test_claim_removal.py` |
| `conversation/` | Session transcript + history-aware query rewriting on top of `Corpus.ask()` | `InMemoryTranscriptStore`, `HistoryAwareQueryRewriter`, `PersonaChatSession` | `test_conversation_chat_session.py`, `test_conversation_transcript.py` |
| `generation/` | Context + query → `Answer` | `ClaudeGenerator` | `test_generation.py`; live test opt-in |
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

### Phase 1 — Ingestion pipeline
- ⬜ Async job queue, dead-letter handling, per-stage checkpointing for
  *ingestion itself* (`Corpus.ingest()` is synchronous, in-process, no queue;
  only the LLM-backed *add-on* passes — concepts/graph/claims/contextual/
  hierarchy — are checkpointed, not ingestion's loader→chunk→embed→store path)
- 🟡 Resumable/idempotent — content-addressed `source_id`/`chunk_id` makes
  re-ingestion idempotent, and `sync_corpus` makes multi-file re-ingest
  diff-aware, but there's no mid-ingestion resume for a single large file
  (loader→chunk→embed→store runs as one call with no stage checkpoint)
- ✅ Content-hash dedupe at file level (`source_id`) and chunk level
  (`chunk_id`); not at page level (no separate page-hash concept)
- 🟡 Parsers behind one interface: PDF, EPUB, DOCX, TXT, MD — ✅; HTML — ⬜
  (no `loaders/html.py`); PDF OCR fallback for scanned pages — ⬜ (PyMuPDF
  text extraction only, no Tesseract/OCR path — confirmed, zero `ocr`
  references in the codebase)
- ✅ Book-aware structure: TOC/chapters/sections, headers/footers/page
  numbers, hyphenation — handled per-loader (not audited line-by-line per
  loader in this pass, but `test_loaders_structure.py` + the P&P fixture
  exercise real chapter/section extraction); multi-column PDF handling and
  figure/table/caption extraction — not verified, likely absent (PyMuPDF
  plain-text extraction mode)
- ✅ Hierarchical chunking with parent-child-ish links — `hierarchy/` gives
  chunk→section→chapter→book; `section_path` breadcrumbs carry structural
  parentage
- ✅ Layout-aware/semantic chunking, pluggable, configurable, overlap,
  token-budget — `StructuralChunker` is the one strategy; it's pluggable
  (the `Chunker` protocol) but no second strategy ships
- ✅ Contextual chunk enrichment (situating blurb before embedding) —
  `contextual/`
- 🟡 Rich metadata schema — `ChunkMeta` has most fields (source_id, title,
  chapter, page range, chunk_id, hash via content-addressing, timestamps via
  `extra`); **no `tenant_id`, no `edition`, no `language`, no explicit
  `ingest_version`** — all phase-7/multi-tenant fields are absent by design
  today, matching Phase 7 being unstarted
- ✅ Multi-file/folder batch upload, progress events, partial-failure
  reporting — `Corpus.ingest_many()` + `IngestProgress` callback; re-ingest
  diffing — `sync_corpus`

### Phase 2 — Embedding and index layer
- 🟡 Provider abstraction — the `Embedder` protocol is real and swappable,
  but **only one provider ships** (Voyage). No OpenAI/Cohere/local
  sentence-transformers/BGE implementation exists. Batching/rate-limit/retry
  — ✅ (per-provider, well done for Voyage). **Cost tracking — ⬜ entirely**
  (no token/dollar accounting anywhere)
- 🟡 Embedding cache keyed by content hash + model version — chunk_id is
  content-addressed and re-embedding is naturally idempotent via upsert, but
  there's no explicit cache keyed on `(hash, model_version)` that would skip
  re-calling the embedder on an unchanged chunk re-ingested under a *new*
  embedder version; versioned embeddings / zero-downtime re-embed migration
  — ⬜
- 🟡 Hybrid indexing: dense ✅ (LanceDB), BM25 ✅ (`lexical/` FTS5), full-text
  — same as BM25 here; **SPLADE — ⬜**
- ⬜ Index lifecycle management: IVF-PQ/HNSW tuning, compaction, per-tenant
  table/partition strategy, benchmarks at 10k/1M/10M chunks — none of this
  exists; `LanceDBStore` uses default LanceDB indexing with no tuning layer.
  Versioned tables / rollback — 🟡 (LanceDB native tag/restore is wired for
  *snapshots*, not for index-tuning rollback)
- ⬜ S3-compatible object storage for raw files, Postgres for
  metadata/jobs/tenancy — neither exists; everything is local SQLite files
  + an on-disk LanceDB directory

### Phase 3 — Retrieval engine
- ✅ Hybrid retrieval with RRF, tunable — `HybridRetriever`
- 🟡 Query understanding — query *rewriting* exists but only inside the
  conversation layer (`HistoryAwareQueryRewriter`, history-aware condensation
  only); **no standalone multi-query expansion, no HyDE, no query
  decomposition into sub-queries** as a retrieval-layer primitive (the
  `IterativeRetriever`'s hop loop is adjacent but is sufficiency-driven
  re-querying, not explicit decomposition)
- ✅ Cross-encoder reranking, pluggable, — `CrossEncoderReranker` +
  `VoyageReranker`; score-threshold cutoff — `Corpus.retrieve(min_score=...)`
- ✅ MMR diversity — `MMRReranker`; metadata filters — `Filter` DSL;
  filter-aware pre-filtering — ✅ (LanceDB `.where()` pre-filters before
  vector search, not post-filtered)
- ⬜ Query router (vector vs keyword vs graph-local vs graph-global vs
  combination, chosen *per query*) — does not exist. The stack is statically
  composed once (`default_stack()`); there is no per-query routing decision
- ✅ Agentic multi-hop retrieval with hard hop budget, stop criterion, full
  trace — `IterativeRetriever` (`max_hops`, `SufficiencyJudge`,
  `last_hops`/`hop_index`/`hop_query` provenance)
- 🟡 Context packing: dedupe/ordering/parent expansion — partially implicit
  (hierarchy/community summaries act as "parent expansion"); **no explicit
  token-budget-aware context packer for a target model's context window** —
  `ClaudeGenerator._format_context` just concatenates everything retrieved

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

1. **Unwrapped backend exceptions throughout the storage layer** (documented
   already in `docs/INTEGRATION_TEST_FINDINGS.md`, re-confirmed here):
   neither `SqliteGraphStore`/`SqliteConceptStore`/`SqliteClaimStore`/
   `SqliteContextStore`/`SqliteHierarchyStore` nor `LanceDBStore` wrap
   backend exceptions (`sqlite3.*`, LanceDB/pyarrow errors) into
   `ragcore.errors.*`. This directly violates `errors.py`'s own stated
   contract ("All library-raised exceptions derive from RagcoreError"). A
   real, reproduced failure mode: losing a SQLite WAL sidecar file raises a
   raw `sqlite3.OperationalError`, not a `StoreError`. This is the single
   highest-value, lowest-risk fix flagged and *not yet made* — a prior
   session explicitly deferred it pending a decision, and no commit since
   addresses it (confirmed: `git log` shows no exception-wrapping commit
   after `c38078c`).
2. **No stage-level resumability inside a single `Corpus.ingest()` call.**
   A large book failing mid-embed re-runs load→chunk from scratch on retry
   (idempotent, but not cheap at scale — no per-stage checkpoint the way
   `build_concepts`/`build_graph`/etc. have).
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
7. **No page-level dedupe**, only file- and chunk-level (spec asks for
   file/page/chunk).
8. **`SourceType` is a closed 5-value `Literal`** (`pdf`/`epub`/`docx`/
   `txt`/`md`) — adding HTML (Phase 1 requirement) is a real, if small,
   breaking-ish change: every `match`/`Literal`-dependent call site that
   assumes exhaustiveness needs auditing, not just a new loader file.
9. **`StructuralChunker` is the only chunking strategy** — "pluggable and
   configurable per document type" (spec) is true at the protocol level but
   not yet demonstrated with a second real strategy.

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

**Phase 1 (ingestion):**
1. HTML loader (closes the stated format list).
2. OCR fallback for scanned PDFs (Tesseract or equivalent behind the
   existing `Loader` interface).
3. Stage-level checkpointing *inside* `Corpus.ingest()` itself (today only
   the add-on LLM passes checkpoint), with dead-letter handling for a
   permanently-failing file in `ingest_many`.
4. Page-level content hash (today: file + chunk only).
5. `tenant_id`/`edition`/`language`/`ingest_version` fields on `ChunkMeta`
   — deferred until Phase 7 design is settled, so these aren't bolted on
   twice.
6. A second `Chunker` strategy (semantic/embedding-boundary chunking) to
   make "pluggable per document type" real, not just protocol-shaped.

**Phase 2 (embedding/index):**
7. Wrap every storage-layer backend exception in `ragcore.errors.StoreError`
   (see §3.1 — highest value, lowest risk, already scoped by a prior
   session).
8. Second embedding provider (OpenAI or local BGE/sentence-transformers) to
   prove the `Embedder` protocol is genuinely swappable, not swappable in
   theory.
9. Cost tracking on every provider call (tokens + $ per request, surfaced
   on the Embedder/Generator, not just retried-and-discarded).
10. Embedding-cache keyed by `(content_hash, model_version)` + a versioned
    re-embedding migration path.
11. Index lifecycle: compaction/tuning hooks on `LanceDBStore`, benchmark
    harness at 10k/1M/10M chunks (today: untested past the 8-book
    integration-suite scale).
12. S3-compatible raw-file storage + Postgres metadata/job store — real
    infrastructure work, sequence after the above, before Phase 7 proper.

**Phase 3 (retrieval):**
13. Standalone query understanding layer: multi-query expansion, HyDE,
    explicit decomposition — independent of the conversation layer's
    history-aware rewriting (which only handles follow-up-turn condensation).
14. Query router (vector/keyword/graph-local/graph-global/combination,
    chosen per query) — the stack is currently static per corpus, not
    adaptive per query.
15. Token-budget-aware context packer for `ClaudeGenerator` (today: naive
    concatenation of every retrieved chunk).

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
