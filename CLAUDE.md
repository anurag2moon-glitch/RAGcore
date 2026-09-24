# ragcore

Reusable retrieval layer for RAG: structure-aware ingestion for long-form
books (PDF, EPUB, DOCX, TXT, MD), a swappable embedder/vector store/generator,
and a GraphRAG layer (entity/relation extraction, graph construction,
graph-aware retrieval) on top. Library, not application — no CLI, no server,
no global state; consuming projects `import ragcore` and drive it. See
`~/.claude/plans/i-m-building-a-reusable-cosmic-quasar.md` for the full
design rationale and decision history.

## `chapter_title` vs `chapter_index`

`ChunkMeta` carries both, and they are **not interchangeable**:

- **`chapter_index`** (`int | None`) — the reliable one. An ordinal over
  structural groups in document order (EPUB spine order, PDF/DOCX page or
  block order). Always clean, always monotonic. **Use this for retrieval
  filtering** (`corpus.retrieve(..., filters={"chapter_index": {"gte": 7}})`).
  Not necessarily the book's own printed chapter number — front matter or
  title-page groups before the first real chapter consume index slots too,
  so "Chapter IV" may land at `chapter_index=5`.
- **`chapter_title`** (`str | None`) — raw, display-only text taken verbatim
  from the source's own structure metadata (EPUB TOC label, PDF outline
  entry, DOCX/MD heading text). **Can be noisy or malformed depending on
  source quality.** Some Project Gutenberg EPUB editions, for example,
  concatenate an illustration caption onto the chapter heading in the TOC
  label itself — a real one we hit: `chapter_title == "I hope Mr. Bingley
  will like it. CHAPTER II."` for what a reader would just call "Chapter 2".
  This isn't a bug in ragcore's extraction — it's literally what the
  source file's TOC says — so don't try to "fix" it with book-specific
  string cleanup in the loader; that just breaks other books' legitimately
  compound titles. If you need a clean per-chapter label for display,
  derive it from `chapter_index` (e.g. "Chapter {chapter_index}") rather
  than trusting `chapter_title` to be clean.
- **`ragcore.stores.lancedb.LanceDBStore`'s Filter DSL enforces this**:
  `chapter_title` is deliberately excluded from the filterable-fields
  allowlist (see `_FILTERABLE_FIELDS` in `ragcore/stores/lancedb.py`) —
  filtering on it raises `StoreError`. Any future `VectorStore` backend
  should keep that restriction.

## GraphRAG: `add_entities`/`add_relations` merge, not replace

Unlike `VectorStore.upsert()` (replace-by-id), every `GraphStore`
implementation must **merge** when a content-addressed entity/relation id
already exists — union `source_chunk_ids` and `extra["aliases"]`, don't
overwrite them. The same fact (e.g. a friendship) is often restated across
chapters; every restatement's chunk should accumulate as provenance. This
is easy to get backwards if you port the vector store's mental model over —
`SqliteGraphStore` (`ragcore/graph/sqlite_store.py`) is the reference
implementation and the thing to compare a new `GraphStore` against.

Entity/relation ids are content-addressed (`entity:{type}:{normalized_name}`,
`relation:{type}:{source_id}:{target_id}`), not random — this is what makes
`build_graph()` idempotent, and lets an exact repeated canonical name collapse
for free (no resolver call at all) the moment it recurs. It does **not**
mean cross-chapter aliasing is rare: tested against real Chapters I-III of
*Pride and Prejudice*, "Bingley" (Ch. I) and "Mr. Bingley" (Ch. II/III) — the
same person, independently canonicalized per chapter — needed
`EmbeddingEntityResolver` to merge, and needed it *again* in Ch. III even
though Ch. II had already resolved it (a fresh extraction call regenerates
the candidate's *own* id from its own chosen name every time, regardless of
what it previously got merged into). The resolver is not a rare-path
across-books-only mechanism — expect it to fire often, even within one book,
whenever per-chapter extraction naming isn't perfectly consistent.

`build_graph()` calls `EntityExtractor.extract()` once per chapter
(`context_group_id`), not per chunk — pronoun/reference resolution ("he",
"his friend") needs chapter-level context, the same reason embeddings are
contextualized at chapter granularity.

## Known limitation: relationally-described entities

`EmbeddingEntityResolver` compares entity *text* (name + aliases, now also a
short relation summary — see below). It has no signal at all for an entity
introduced only by relational description, never by name. Real case found
testing against Ch. III: Darcy/Bingley's ballroom exchange refers to Jane
only as **"the eldest Miss Bennet"** — a Regency courtesy-title convention
(the eldest unmarried daughter is "Miss Surname", no first name) — while
"Jane" as a name only appears back in Chapter I. Per-chapter extraction
produces two separate candidates; nothing in the resolver's text-similarity
comparison connects "the eldest Miss Bennet" to "Jane" — there's no shared
vocabulary, only plot knowledge a pure text embedder doesn't have.

**Mitigation shipped:** `ragcore.graph.pipeline.build_graph` now attaches a
short relation summary to each candidate before resolution — e.g. "the
eldest Miss Bennet" gets `extra["relations"] = ["danced_with Mr. Bingley"]`
— and `_entity_text` (`ragcore/graph/resolver.py`) folds it into the
embedded text. `SqliteGraphStore._merge_extra` accumulates it across merges
the same way it does aliases. This is a **partial** fix: it only helps when
the two candidates happen to share a relation to an entity that's *already*
linked under the same id on both sides. It would not have fixed the Jane
case above on its own — Jane's Chapter I mention carries no relation at all
in our test fixture, so there's nothing to overlap with "danced_with Mr.
Bingley". It helps a real but narrower slice of cases than the general
problem.

**Why the full mechanism (cardinality-aware relational resolution) wasn't
built:** the natural extension — treat entities sharing relations as likely
duplicates — is actively dangerous for exactly the family-relation case
that motivates it. Every Bennet daughter (Jane, Elizabeth, Kitty, Lydia,
Mary) shares `parent_of ← Mr. Bennet`, attends the same ball, gets discussed
in the same conversations — naive relational similarity would tend to merge
*siblings*, not resolve aliases. Making it safe requires: (1) an extraction
schema with **cardinality-aware relation types** (`eldest_daughter_of`, not
generic `parent_of` — a type where at most one entity can hold it for a
given parent), (2) resolver logic that recognizes which relation types are
functionally unique and treats a collision on those as a near-deterministic
merge signal, and (3) in general, actually *establishing* the qualifying
fact (e.g. Jane's birth order is never stated outright near her Ch. I
mention — nothing in this fixture's relations would let you infer "Jane is
eldest" even with cardinality reasoning). That's a real feature, not a
tweak, and the failure mode is more a 19th-century-prose convention
(courtesy titles, epithets) than something expected in a nonfiction corpus
whose concepts are named directly and consistently. Revisit only if real
usage on the actual target book(s) shows this costing retrieval/graph
quality often enough to justify the complexity — not pre-emptively.

## Concept layer: build_concepts() is a separate checkpointed pass, not part of ingest()

`ragcore/concepts/` extracts abstract concepts/themes (not keywords — a
*proposition*, e.g. "consequences of unchecked ambition", never "ambition")
and their co-occurrence edges, one LLM call per chapter, into a
`ConceptStore`. Like `build_graph()`, `build_concepts()` runs **after**
`corpus.ingest()`, never inside it — `ingest(replace=True)` has already
deleted the old chunks before `store.upsert()` writes new ones, so an
in-ingest LLM failure either runs before chunks exist (dangling edges) or
after (a real extraction failure gets misreported as a failed ingest even
though the chunks landed fine). Content-addressed `chunk_id`s make the split
free: re-ingesting the same file reproduces identical ids, so a later
`build_concepts()` run against them is exactly as valid as `build_graph()`
already is.

**Resumability**: unlike `build_graph()`, `build_concepts()` is checkpointed
— a concept corpus can be far larger than one book's worth of entity
extraction, and losing partial progress on a multi-hour ingestion is a real
cost. The resume unit is `(source_id, context_group_id)` — one chapter,
matching the extraction call granularity. `SqliteConceptStore` writes a
chapter's concepts, edges, *and* its checkpoint row inside **one**
transaction (`add_chapter_concepts`) — this is the one place its pattern
diverges from `SqliteGraphStore`, which uses two separate transactions for
`add_entities`/`add_relations` and gets away with it only because
`build_graph()` has no checkpoint to leave inconsistent. A checkpoint's
fingerprint is `sha256(model | prompt_version | schema_version | sorted
chunk_ids)`, so a model change, a prompt-version bump, or a chunker change
that shifts chunk boundaries (chunk_ids are themselves content-addressed)
all invalidate it automatically. **The LLM call always runs before the
transaction opens, never inside it** — holding SQLite's write lock for the
duration of an API call would starve every other writer up to
`busy_timeout` (30s).

**Edge weight is never an incrementally-updated counter.** Two concepts
co-occurring in a chunk get one row in `concept_edge_chunks` (INSERT OR
IGNORE — a plain idempotent set union, same as `concept_chunks`/
`entity_chunks`), and weight is derived as `COUNT(DISTINCT chunk_id)` when
an edge is read, never accumulated on write. An incrementing counter would
double-count on a re-run of the same chapter (e.g. after a fingerprint
bump), breaking the idempotency every other write in this file relies on —
this was caught by `test_concept_store.py`'s idempotency test before it
shipped, not discovered after.

Community detection (`ragcore/concepts/communities.py`, Louvain via
`networkx`, the `concepts` extra) is checkpointed separately, at
corpus-global granularity — a fingerprint over every concept id and every
`(source, target, weight)` edge triple, so adding or resolving even one
concept invalidates it and the next call reclusters. Community id is
content-addressed over sorted member concept ids
(`community:{sha256(...)[:16]}`), which is what makes summary recomputation
resumable: an unchanged community keeps its id and its existing
`CommunitySummary` is left alone.

**Community summaries reach retrieval as synthetic `Chunk`s**, not through a
new field on `Retrieved` or a change to the `Generator` protocol. See
`ChunkOrigin` in `ragcore/types.py` (`SourceType | Literal["synthetic"]` —
deliberately *not* added to `SourceType` itself, since no `Loader` ever
produces one) and `ragcore.retrieval.concept._summary_chunk`. A synthetic
chunk's `chunk_id`/`source_id` is the community's own id (already namespaced
as `community:{digest}` — do not re-prefix it, `_summary_chunk` doesn't),
and `extra["ragcore_kind"] == "community_summary"` marks it for
`MMRReranker` (exempt from the per-source cap, bounded by `max_summaries`
instead) and `ClaudeVerifier` (a claim supported only by chunks with this
marker gets `status="supported_by_synthesis"`, derived mechanically from
provenance — never trusted to the model's own self-classification).
`LanceDBStore.upsert()` rejects `source_type == "synthetic"` outright: a
summary is generated at retrieval time and must never be persisted as if it
were real source material.

Every concept's embedding and every community summary's embedding are
persisted (`Concept.vector`, `CommunitySummary.vector`), not recomputed per
query — this is what lets `EmbeddingConceptMapper` (the default
`ConceptMapper`, no LLM call) and `ConceptRetriever`'s scoring of
concept-linked chunks work without hitting the embedder on every lookup.
`ConceptRetriever` scores concept/summary hits with a real query-vector
cosine under the identical `1/(1+distance)` transform
`LanceDBStore.query()` uses, specifically so they sort correctly against
`GraphRetriever`'s decayed expansion scores and plain vector hits in one
merged, `MMRReranker`-rerankable list — an invented constant score would
have made that ordering meaningless the moment more than one expansion tier
was in play.

**Cost shape**: the default stack (`EmbeddingConceptMapper`, no
`VerifyingGenerator` wired) adds retrieval-time embedding calls only —
`Corpus.ask()` stays one generation call, same as today. `ClaudeConceptMapper`
(wraps an `EmbeddingConceptMapper`'s shortlist, refines it with one
`messages.parse` call) adds one call per query; `VerifyingGenerator`
(`corpus.generator = VerifyingGenerator(corpus.generator, ClaudeVerifier())`)
adds two per `ask()` (decompose, then check). Both are opt-in decorators —
wiring nothing new leaves `ask()`'s cost and behavior identical to before
this layer existed.

## Hybrid retrieval, contextual embeddings, eval harness, multi-hop, cross-encoder, hierarchy, incremental ingestion

Seven more layers, added in one pass, following the same conventions as the
concept layer above. **Zero new PyPI dependencies** across all seven — see
each item for why.

- **Hybrid retrieval** (`ragcore/lexical/`, `ragcore/retrieval/hybrid.py`):
  `Fts5LexicalStore` is SQLite FTS5 via stdlib `sqlite3` (no new dep — same
  pattern as `SqliteGraphStore`), fused with vector hits via reciprocal rank
  fusion in `HybridRetriever`. **Critical invariant**: RRF decides the
  candidate *set*; it never decides a result's `score`. Raw RRF scores
  (~0.016) are on a totally different scale than the `1/(1+distance)` cosine
  scale every other layer assumes — `HybridRetriever` re-scores every RRF
  winner with `ragcore.retrieval.scoring.cosine_score()` against its real
  vector before returning it. `HybridRetriever` is a peer of
  `VectorRetriever` (the new base of the stack), not a decorator around it.
  `build_lexical_index()` is a separate, checkpoint-free pass (indexing is
  idempotent and has no LLM call, so it needs none) — `Corpus.ingest()`
  itself stays untouched, same as every other add-on layer.

- **Contextual chunk embeddings** (`ragcore/contextual/`): a separate
  checkpointed pass, `build_contextual_embeddings()`, mirroring
  `build_concepts()` exactly. Writes a 1-2 sentence blurb per chunk (one LLM
  call per chapter), embeds `f"{blurb}\n\n{chunk.text}"`, and re-upserts
  under the same `chunk_id`. **The blurb never touches `Chunk.text`** —
  only the string handed to the embedder and `meta.extra["context_blurb"]`
  — so citations and `ClaudeVerifier` never see contextualization
  scaffolding presented as "the source passage."

- **Evaluation harness** (`ragcore/eval/`): a JSONL gold-question format
  (`GoldQuestion`/`GoldPassage`, matched against retrieved chunks at three
  granularities — exact `chunk_id`, then `(source_id, chapter_index)`, then
  `source_path` alone — so a re-chunked corpus doesn't silently zero out an
  old gold set), `run_eval()` for recall/precision/nDCG/MRR, and an
  **opt-in** groundedness score reusing `ClaudeVerifier` directly (omit
  `generator`/`verifier` for free retrieval-only metrics). Distinct from,
  and does not touch, `tests/integration/eval/` — that harness stays
  test-only, coupled to synthetic-library fixtures, per its own docstring;
  this is a general-purpose tool for a consumer's own corpus. Starter set
  and authoring helper (`find_gold_passage`, resolves a marker substring to
  a real chunk) in `ragcore/eval/examples/starter.jsonl`.

- **Multi-hop iterative retrieval** (`ragcore/retrieval/iterative.py`):
  `IterativeRetriever` loops `retrieve()` internally — hop 0 with the
  original query, then a `SufficiencyJudge` (default `ClaudeSufficiencyJudge`)
  decides whether to stop or continue with a refined sub-query, up to
  `max_hops` (default 3). Fits the `Retriever` protocol with no changes to
  `Corpus`. Hop provenance is stamped into
  `chunk.meta.extra["hop_index"]`/`["hop_query"]` and also readable from
  `IterativeRetriever.last_hops` after a call. On a chunk re-found across
  hops: keeps the *earliest* hop's stamp, but the *higher* of the two
  scores.

- **Cross-encoder reranking** (`ragcore/reranking/`,
  `ragcore/retrieval/cross_encoder.py`): `VoyageReranker` uses
  `voyageai.Client.rerank()` — already in the required `voyageai` package,
  zero new dependency (deliberately not a local `sentence-transformers`/
  torch model). **Runs before `MMRReranker`, not after**: MMR diversifies an
  *already relevance-ranked* set, so reranking after MMR would only reorder
  display within whatever set MMR already picked, never influence *which*
  documents were picked. Stamps `extra["cross_encoder_score"]` without
  touching `Retrieved.score` (so `Corpus.retrieve(min_score=...)`'s meaning
  never silently changes). `MMRReranker` now prefers
  `cross_encoder_score` over `.score` when *every* pool candidate has one —
  all-or-nothing per query, never a mixed-scale min-max normalization.

- **Hierarchical summarization** (`ragcore/hierarchy/`,
  `ragcore/retrieval/hierarchy.py`): chunk → section → chapter → book, via
  `build_hierarchy()` — another separate checkpointed pass, using only
  existing `ChunkMeta.section_path`/`chapter_index`/`section` metadata (no
  chunker changes). A chapter with no distinct sub-sections (every chunk
  sharing one `section_path`) skips straight to chapter-level
  summarization. Node ids are **positional**
  (`hierarchy:chapter:{source_id}:{chapter_index}`), not content-addressed
  like concepts/communities — a hierarchy node's identity is inherently
  tied to one position in one book, unlike a concept that can legitimately
  recur across sources. `HierarchyRetriever` surfaces relevant nodes as
  synthetic chunks, same as community summaries — **except a hierarchy
  summary carries the real book's `source_id`** (only `source_type` stays
  `"synthetic"`) and is deliberately **not** exempt from `MMRReranker`'s
  per-source cap, since it competes with that same book's own chunks for
  context budget. `MMRReranker.summary_caps` (a `dict[str, int]`, default
  `{"community_summary": 2, "hierarchy_summary": 2}`) replaced the old
  single `max_summaries` int, which is kept as a backward-compatible alias.

- **Incremental ingestion with versioning** (`ragcore/versioning/`):
  `IngestionManifest` (own SQLite file) tracks path → source_id/content-hash
  history — state `Corpus` deliberately never keeps. `plan_sync()` diffs a
  given path list against it (added/changed/removed/unchanged, by
  re-hashing with the same sha256 scheme `Document.source_id` already
  uses). `sync_corpus()` orchestrates: snapshot (if enabled) → prune
  removed/changed sources from *every* provided store, before any rebuild →
  ingest new/changed paths → re-run every LLM-backed pass **scoped to only
  the new/changed source_ids** → re-run the corpus-global concept-community
  passes unscoped (already self-checkpointing). **A real latent
  characteristic of `Corpus.ingest()` this surfaces, not one it
  introduces**: since `source_id` is a content hash, a changed file
  computes a brand-new one, so `ingest(replace=True)`'s
  `store.delete(source_id=doc.source_id)` deletes nothing under the *new*
  hash and the file's *old* chunks are silently orphaned — true of a bare
  `corpus.ingest()` call outside `sync_corpus` too. `sync_corpus` is what
  looks up the *old* source_id by path and prunes it explicitly.
  `SqliteGraphStore`/`SqliteConceptStore` gained their first-ever removal
  methods (`remove_chunk_ids`/`remove_source`) for this — previously purely
  additive. Rollback: `LanceDBStore.snapshot()`/`restore_snapshot()` use
  LanceDB's own native tag/restore API (append-only — a rollback is itself
  a new version, never destroys history); every SQLite store gained
  `backup_to()`/`restore_from_backup()` using the **stdlib sqlite3 online
  backup API in both directions** — never `shutil.copy`, which can capture
  a torn snapshot of a live WAL-mode database file. A snapshot is recorded
  `'pending'` before either half runs and flipped to `'complete'` only
  after both succeed; `rollback()` refuses a `'pending'` snapshot.

`ragcore/retrieval/stack.py::default_stack(corpus, ...)` wires the full
default retriever stack in one place (only including the layers whose
backing store you actually pass):

```
MMRReranker(
  CrossEncoderReranker(
    IterativeRetriever(
      HierarchyRetriever(
        ConceptRetriever(
          GraphRetriever(
            HybridRetriever(embedder, store, lexical, corpus))))))))
```

`IterativeRetriever` is opt-in (`iterative=True` or pass a `judge`) even
inside `default_stack()`, since its default `ClaudeSufficiencyJudge`
eagerly constructs an `anthropic.Anthropic()` client.

## Local dev

```
source .venv/bin/activate
pytest -q                    # unit tests, no API keys, ~1s
pytest -q -m live            # + live tests against real Voyage/Claude APIs
python scripts/query.py samples/library.lancedb "your question" [k] [--ask]
python scripts/concepts_demo.py    # concept-layer wiring demo, no API calls
python scripts/run_eval.py --demo  # eval harness demo, no API calls
```

`VOYAGE_API_KEY` is required for anything that embeds (ingestion, or any
`retrieve()`/`ask()` call, since queries are embedded too). `ANTHROPIC_API_KEY`
is required for `--ask` / `Corpus.ask()`, and additionally gates concept
extraction, community summarization, `ClaudeConceptMapper`,
`ClaudeVerifier`, contextual-embedding blurb writing,
`ClaudeSufficiencyJudge`, and `ClaudeHierarchySummarizer`. Cross-encoder
reranking (`VoyageReranker`) reuses `VOYAGE_API_KEY`, not a separate key.

`samples/` (gitignored) holds test corpora — not part of the package.
