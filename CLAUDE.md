# ragcore

Reusable retrieval layer for RAG: structure-aware ingestion for long-form
books (PDF, EPUB, DOCX, TXT, MD), a swappable embedder/vector store/generator,
and a GraphRAG layer designed in but not yet built. Library, not application —
no CLI, no server, no global state; consuming projects `import ragcore` and
drive it. See `~/.claude/plans/i-m-building-a-reusable-cosmic-quasar.md` for
the full design rationale and decision history.

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
