"""Requirement 1: ingest many full-length books, of varying structure, into
one shared index; verify the index stays internally consistent at that
scale, not just for the single-book case the existing unit tests cover.
"""

from __future__ import annotations

from ragcore.corpus import Corpus
from tests.integration.conftest import SharedLibrary
from tests.integration.synth.library import Library


def test_all_books_ingested_with_no_errors(shared_library: SharedLibrary) -> None:
    sources = shared_library.corpus.sources()
    assert len(sources) == len(shared_library.book_keys)
    assert {s.source_id for s in sources}  # every source got a real id
    total = sum(s.chunk_count for s in sources)
    assert total == shared_library.corpus.count()
    assert total > 0


def test_chunk_ids_globally_unique_across_the_whole_library(shared_library: SharedLibrary) -> None:
    corpus = shared_library.corpus
    all_ids: list[str] = []
    for source in corpus.sources():
        chunks = corpus.store.get_by_source(source.source_id)
        all_ids.extend(c.chunk_id for c in chunks)
    assert len(all_ids) == len(set(all_ids)), "duplicate chunk_id across sources"
    assert len(all_ids) == corpus.count()


def test_no_cross_source_metadata_bleed(shared_library: SharedLibrary) -> None:
    """Every chunk fetched under a given source_id must actually carry
    that source_id (and a matching source_path) in its own metadata — a
    shared vector table across many sources is exactly where a filter or
    join bug would leak one book's chunks into another's results."""
    corpus = shared_library.corpus
    for source in corpus.sources():
        chunks = corpus.store.get_by_source(source.source_id)
        assert chunks, f"no chunks for {source.source_id}"
        for c in chunks:
            assert c.meta.source_id == source.source_id
            assert c.meta.source_path == source.path


def test_chapter_index_monotonic_within_each_source(shared_library: SharedLibrary) -> None:
    """chapter_index is documented as a clean, monotonic ordinal over
    structural groups in document order (CLAUDE.md) — verify that holds at
    library scale, across every structured-format book, not just the one
    fixture book the design doc's example was written against."""
    corpus = shared_library.corpus
    checked_any = False
    for source in corpus.sources():
        chunks = sorted(corpus.store.get_by_source(source.source_id), key=lambda c: c.meta.chunk_index)
        indices = [c.meta.chapter_index for c in chunks if c.meta.chapter_index is not None]
        if not indices:
            continue  # e.g. the no_toc book: no structure at all
        checked_any = True
        assert indices == sorted(indices), f"{source.path}: chapter_index not monotonic"
    assert checked_any


def test_section_path_survives_ingestion(shared_library: SharedLibrary) -> None:
    corpus = shared_library.corpus
    any_with_sections = False
    for source in corpus.sources():
        for c in corpus.store.get_by_source(source.source_id):
            if c.meta.section_path:
                any_with_sections = True
                assert all(isinstance(p, str) for p in c.meta.section_path)
    assert any_with_sections


def test_reingest_same_book_is_idempotent(fresh_corpus: Corpus, synth_library: Library) -> None:
    spec = synth_library.book("clean_toc")
    r1 = fresh_corpus.ingest(spec.path)
    ids_1 = {c.chunk_id for c in fresh_corpus.store.get_by_source(r1.source_ids[0])}
    count_1 = fresh_corpus.count()

    r2 = fresh_corpus.ingest(spec.path)  # replace=True by default
    ids_2 = {c.chunk_id for c in fresh_corpus.store.get_by_source(r2.source_ids[0])}

    assert r1.source_ids == r2.source_ids  # same content-addressed source_id
    assert ids_1 == ids_2
    assert fresh_corpus.count() == count_1
    assert r2.sources_replaced == r1.source_ids


def test_remove_one_source_leaves_others_untouched(fresh_corpus: Corpus, synth_library: Library) -> None:
    keys = ("very_short", "unusual_naming", "clean_toc")
    reports = {k: fresh_corpus.ingest(synth_library.book(k).path) for k in keys}
    total_before = fresh_corpus.count()

    victim = "unusual_naming"
    removed = fresh_corpus.remove(reports[victim].source_ids[0])

    assert removed > 0
    assert fresh_corpus.count() == total_before - removed
    remaining_sources = {s.source_id for s in fresh_corpus.sources()}
    assert reports[victim].source_ids[0] not in remaining_sources
    for k in keys:
        if k == victim:
            continue
        assert reports[k].source_ids[0] in remaining_sources
        # every surviving source's chunks are all still fetchable and intact
        chunks = fresh_corpus.store.get_by_source(reports[k].source_ids[0])
        assert len(chunks) == reports[k].chunks_ingested


def test_get_with_large_id_list_across_many_sources(shared_library: SharedLibrary) -> None:
    """Locks in verified-good LanceDBStore behavior: a large IN(...) id
    list (spanning many sources, not just one) round-trips correctly."""
    corpus = shared_library.corpus
    all_ids: list[str] = []
    for source in corpus.sources():
        all_ids.extend(c.chunk_id for c in corpus.store.get_by_source(source.source_id))
    sample = all_ids[: min(2000, len(all_ids))]
    fetched = corpus.get(sample)
    assert {c.chunk_id for c in fetched} == set(sample)
