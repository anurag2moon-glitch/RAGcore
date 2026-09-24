"""Opt-in layer: real Project Gutenberg books (see
scripts/fetch_gutenberg.py) read from a local, gitignored cache.

Real EPUBs are the actual "unusual/messy structure" ground truth — the
synthetic library (tests/integration/synth) is authored to *mimic* known
real-world pathologies (a Gutenberg-style noisy TOC-label caption, several
chapters packed into one XHTML file), but only a real download proves the
loader survives the genuine, un-curated thing. There's no planted ground
truth here (no needles, no known-relevant chunk ids), since these are real
books nobody hand-labeled — assertions are structural-survival checks
(loads without crashing, produces plausible chunk counts, chapter_index
stays clean and monotonic even when chapter_title is noisy), not retrieval
quality checks.

Skips cleanly (not an error) when the cache is empty — CI and any
machine that hasn't run `python scripts/fetch_gutenberg.py` are
unaffected. Every test here carries `@pytest.mark.gutenberg`, excluded
from the default `pytest -m 'not live and not gutenberg'` run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragcore.corpus import Corpus
from tests.integration.doubles import DeterministicEmbedder
from tests.fakes import InMemoryStore

CACHE_DIR = Path(__file__).parent / ".gutenberg_cache"

pytestmark = pytest.mark.gutenberg


def _cached_books() -> list[Path]:
    if not CACHE_DIR.exists():
        return []
    return sorted(CACHE_DIR.glob("*.epub"))


@pytest.fixture(scope="module")
def gutenberg_corpus() -> Corpus:
    books = _cached_books()
    if not books:
        pytest.skip(
            "no cached Gutenberg books found — run `python scripts/fetch_gutenberg.py` first"
        )
    corpus = Corpus(InMemoryStore(dim=128), DeterministicEmbedder(dim=128))
    for path in books:
        corpus.ingest(path)
    return corpus


def test_all_cached_books_ingest_without_error(gutenberg_corpus: Corpus) -> None:
    sources = gutenberg_corpus.sources()
    assert len(sources) == len(_cached_books())
    assert gutenberg_corpus.count() > 0
    for s in sources:
        assert s.chunk_count > 0, f"{s.path}: ingested with zero chunks"


def test_chapter_index_stays_clean_even_where_chapter_title_is_noisy(gutenberg_corpus: Corpus) -> None:
    """The documented real pathology (CLAUDE.md): some Project Gutenberg
    EPUB editions concatenate an illustration caption onto the chapter
    heading in the TOC label itself. chapter_title can be noisy;
    chapter_index must stay a clean, monotonic ordinal regardless."""
    checked_any = False
    for source in gutenberg_corpus.sources():
        chunks = sorted(
            gutenberg_corpus.store.get_by_source(source.source_id), key=lambda c: c.meta.chunk_index
        )
        indices = [c.meta.chapter_index for c in chunks if c.meta.chapter_index is not None]
        if not indices:
            continue
        checked_any = True
        assert indices == sorted(indices), f"{source.path}: chapter_index not monotonic"
        # chapter_title is display-only and allowed to be messy — just
        # confirm every chunk that has one carries a real string, never
        # None-where-a-title-should-be silently swallowed into an
        # exception, and never used for anything but display.
        for c in chunks:
            if c.meta.chapter_title is not None:
                assert isinstance(c.meta.chapter_title, str)
    assert checked_any


def test_reingesting_a_real_book_is_idempotent(gutenberg_corpus: Corpus) -> None:
    books = _cached_books()
    target = books[0]
    corpus = Corpus(InMemoryStore(dim=64), DeterministicEmbedder(dim=64))
    r1 = corpus.ingest(target)
    ids_1 = {c.chunk_id for c in corpus.store.get_by_source(r1.source_ids[0])}
    r2 = corpus.ingest(target)
    ids_2 = {c.chunk_id for c in corpus.store.get_by_source(r2.source_ids[0])}
    assert r1.source_ids == r2.source_ids
    assert ids_1 == ids_2


def test_retrieval_against_real_books_returns_well_formed_results(gutenberg_corpus: Corpus) -> None:
    for query in ("love and marriage", "a mysterious letter", "the strange transformation", "revenge"):
        results = gutenberg_corpus.retrieve(query, k=10)
        assert len(results) <= 10
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
