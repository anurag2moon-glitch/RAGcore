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

## Local dev

```
source .venv/bin/activate
pytest -q                    # unit tests, no API keys, ~1s
pytest -q -m live            # + live tests against real Voyage/Claude APIs
python scripts/query.py samples/library.lancedb "your question" [k] [--ask]
```

`VOYAGE_API_KEY` is required for anything that embeds (ingestion, or any
`retrieve()`/`ask()` call, since queries are embedded too). `ANTHROPIC_API_KEY`
is only required for `--ask` / `Corpus.ask()`.

`samples/` (gitignored) holds test corpora — not part of the package.
