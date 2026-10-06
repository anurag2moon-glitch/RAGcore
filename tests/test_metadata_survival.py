"""End-to-end pipeline test: load -> chunk -> embed -> store -> retrieve,
using FakeEmbedder + InMemoryStore so it runs with no network and no
native deps. This is the load-bearing test from the design doc's
verification plan — it proves structure metadata survives the whole
pipeline, not just extraction.
"""

from __future__ import annotations

from pathlib import Path

from ragcore.corpus import Corpus
from tests.conftest import CH1_TITLE, CH2_TITLE, SEC_TITLE
from tests.fakes import FakeEmbedder, InMemoryStore


def _build_corpus(dim: int = 8) -> tuple[Corpus, InMemoryStore]:
    store = InMemoryStore(dim=dim)
    corpus = Corpus(store, FakeEmbedder(dim=dim))
    return corpus, store


def test_metadata_survives_full_pipeline(md_book: Path) -> None:
    corpus, store = _build_corpus()

    report = corpus.ingest(md_book, metadata={"project": "test-suite"})
    assert report.chunks_ingested > 0
    assert corpus.count() == report.chunks_ingested

    chunks = store.all_chunks()
    by_chapter: dict[str | None, list] = {}
    for c in chunks:
        by_chapter.setdefault(c.meta.chapter_title, []).append(c)

    assert CH1_TITLE in by_chapter
    assert CH2_TITLE in by_chapter
    assert any(c.meta.section == SEC_TITLE for c in by_chapter[CH1_TITLE])

    # chapter_index is stable and ordinal across chapters
    assert {c.meta.chapter_index for c in by_chapter[CH1_TITLE]} == {0}
    assert {c.meta.chapter_index for c in by_chapter[CH2_TITLE]} == {1}

    # caller-supplied metadata merged into extra, source fields preserved
    assert all(c.meta.extra.get("project") == "test-suite" for c in chunks)
    assert all(c.meta.source_type == "md" for c in chunks)
    assert all(c.meta.source_path == str(md_book) for c in chunks)
    assert all(c.meta.source_id == chunks[0].meta.source_id for c in chunks)


def test_retrieved_chunks_carry_full_metadata(md_book: Path) -> None:
    corpus, _ = _build_corpus()
    corpus.ingest(md_book)

    results = corpus.retrieve("background on chunking and embeddings", k=10)
    assert results
    for r in results:
        assert r.chunk.meta.source_type == "md"
        assert r.chunk.meta.chapter_title is not None
        assert r.graph is None  # graph layer not attached


def test_filters_restrict_results_by_structure(md_book: Path) -> None:
    corpus, store = _build_corpus()
    corpus.ingest(md_book)

    ch2_ids = {c.chunk_id for c in store.all_chunks() if c.meta.chapter_title == CH2_TITLE}
    filtered = corpus.retrieve("chunking", k=50, filters={"chapter_index": {"gte": 1}})

    assert filtered
    assert all(r.chunk.meta.chapter_index is not None and r.chunk.meta.chapter_index >= 1 for r in filtered)
    assert {r.chunk.chunk_id for r in filtered} <= ch2_ids


def test_tenant_id_defaults_to_none_when_not_passed(md_book: Path) -> None:
    corpus, store = _build_corpus()
    corpus.ingest(md_book)

    assert all(c.meta.tenant_id is None for c in store.all_chunks())


def test_ingest_tenant_id_tags_every_chunk_from_that_file(md_book: Path) -> None:
    corpus, store = _build_corpus()
    corpus.ingest(md_book, tenant_id="tenant-a")

    chunks = store.all_chunks()
    assert chunks
    assert all(c.meta.tenant_id == "tenant-a" for c in chunks)


def test_retrieve_filters_by_tenant_id_isolate_tenants(md_book: Path, tmp_path: Path) -> None:
    # Distinct content for book_b, not a copy of md_book: source_id is a
    # content hash (sha256 of file bytes), so two files with identical
    # bytes collapse to the same source_id and the second ingest()'s
    # default replace=True would delete the first tenant's chunks.
    corpus, _store = _build_corpus()
    book_b = tmp_path / "book_b.md"
    book_b.write_text(md_book.read_text(encoding="utf-8") + "\n\nExtra unique text for book b.\n", encoding="utf-8")

    corpus.ingest(md_book, tenant_id="tenant-a")
    corpus.ingest(book_b, tenant_id="tenant-b")

    results = corpus.retrieve("chunking", k=50, filters={"tenant_id": "tenant-a"})

    assert results
    assert all(r.chunk.meta.tenant_id == "tenant-a" for r in results)
