"""TenantScopedCorpus: enforced tenant_id on every ingest/retrieve/ask/
get/sources/remove/count call, not just made possible by a filter the
caller has to remember (docs/adr/0006-tenant-id-foundation-not-enforcement.md).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragcore.corpus import Corpus
from ragcore.errors import TenancyError
from ragcore.tenancy import TenantScopedCorpus
from tests.fakes import FakeEmbedder, InMemoryStore


def _two_tenant_corpus(md_book: Path, tmp_path: Path) -> tuple[Corpus, Path, Path]:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    book_b = tmp_path / "book_b.md"
    book_b.write_text(
        md_book.read_text(encoding="utf-8") + "\n\nExtra unique text for book b.\n",
        encoding="utf-8",
    )
    return corpus, md_book, book_b


def test_rejects_an_empty_tenant_id() -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    with pytest.raises(TenancyError):
        TenantScopedCorpus(corpus, "")


def test_ingest_tags_every_chunk_with_the_bound_tenant_id(md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    scoped = TenantScopedCorpus(corpus, "tenant-a")

    scoped.ingest(md_book)

    chunks = corpus.store.all_chunks()
    assert chunks
    assert all(c.meta.tenant_id == "tenant-a" for c in chunks)


def test_ingest_raises_on_a_conflicting_explicit_tenant_id(md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    scoped = TenantScopedCorpus(corpus, "tenant-a")

    with pytest.raises(TenancyError):
        scoped.ingest(md_book, tenant_id="tenant-b")


def test_ingest_allows_an_explicit_tenant_id_that_matches(md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    scoped = TenantScopedCorpus(corpus, "tenant-a")

    report = scoped.ingest(md_book, tenant_id="tenant-a")

    assert report.chunks_ingested > 0


def test_retrieve_never_sees_another_tenants_chunks(md_book: Path, tmp_path: Path) -> None:
    corpus, book_a, book_b = _two_tenant_corpus(md_book, tmp_path)
    TenantScopedCorpus(corpus, "tenant-a").ingest(book_a)
    TenantScopedCorpus(corpus, "tenant-b").ingest(book_b)

    results = TenantScopedCorpus(corpus, "tenant-a").retrieve("chunking", k=50)

    assert results
    assert all(r.chunk.meta.tenant_id == "tenant-a" for r in results)


def test_retrieve_raises_on_a_conflicting_filter(md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    scoped = TenantScopedCorpus(corpus, "tenant-a")
    scoped.ingest(md_book)

    with pytest.raises(TenancyError):
        scoped.retrieve("chunking", filters={"tenant_id": "tenant-b"})


def test_retrieve_allows_a_non_conflicting_filter_merged_with_tenant_id(md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    scoped = TenantScopedCorpus(corpus, "tenant-a")
    scoped.ingest(md_book)

    results = scoped.retrieve("chunking", k=50, filters={"chapter_index": {"gte": 0}})

    assert results
    assert all(r.chunk.meta.tenant_id == "tenant-a" for r in results)


def test_get_raises_when_a_chunk_belongs_to_another_tenant(md_book: Path, tmp_path: Path) -> None:
    corpus, book_a, book_b = _two_tenant_corpus(md_book, tmp_path)
    TenantScopedCorpus(corpus, "tenant-a").ingest(book_a)
    TenantScopedCorpus(corpus, "tenant-b").ingest(book_b)

    foreign_chunk_id = next(c.chunk_id for c in corpus.store.all_chunks() if c.meta.tenant_id == "tenant-b")

    with pytest.raises(TenancyError):
        TenantScopedCorpus(corpus, "tenant-a").get([foreign_chunk_id])


def test_get_succeeds_for_this_tenants_own_chunk(md_book: Path, tmp_path: Path) -> None:
    corpus, book_a, book_b = _two_tenant_corpus(md_book, tmp_path)
    TenantScopedCorpus(corpus, "tenant-a").ingest(book_a)
    TenantScopedCorpus(corpus, "tenant-b").ingest(book_b)

    own_chunk_id = next(c.chunk_id for c in corpus.store.all_chunks() if c.meta.tenant_id == "tenant-a")

    [chunk] = TenantScopedCorpus(corpus, "tenant-a").get([own_chunk_id])
    assert chunk.chunk_id == own_chunk_id


def test_sources_only_lists_this_tenants_own_sources(md_book: Path, tmp_path: Path) -> None:
    corpus, book_a, book_b = _two_tenant_corpus(md_book, tmp_path)
    TenantScopedCorpus(corpus, "tenant-a").ingest(book_a)
    TenantScopedCorpus(corpus, "tenant-b").ingest(book_b)

    sources_a = TenantScopedCorpus(corpus, "tenant-a").sources()

    assert len(sources_a) == 1
    assert sources_a[0].path == str(book_a)


def test_count_only_counts_this_tenants_own_chunks(md_book: Path, tmp_path: Path) -> None:
    corpus, book_a, book_b = _two_tenant_corpus(md_book, tmp_path)
    scoped_a = TenantScopedCorpus(corpus, "tenant-a")
    scoped_b = TenantScopedCorpus(corpus, "tenant-b")
    scoped_a.ingest(book_a)
    scoped_b.ingest(book_b)

    assert scoped_a.count() == len([c for c in corpus.store.all_chunks() if c.meta.tenant_id == "tenant-a"])
    assert scoped_a.count() + scoped_b.count() == corpus.count()


def test_remove_raises_when_the_source_belongs_to_another_tenant(md_book: Path, tmp_path: Path) -> None:
    corpus, book_a, book_b = _two_tenant_corpus(md_book, tmp_path)
    TenantScopedCorpus(corpus, "tenant-a").ingest(book_a)
    report_b = TenantScopedCorpus(corpus, "tenant-b").ingest(book_b)

    with pytest.raises(TenancyError):
        TenantScopedCorpus(corpus, "tenant-a").remove(report_b.source_ids[0])


def test_remove_returns_zero_for_a_nonexistent_source(md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    scoped = TenantScopedCorpus(corpus, "tenant-a")
    scoped.ingest(md_book)

    assert scoped.remove("nonexistent-source-id") == 0


def test_remove_deletes_this_tenants_own_source(md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    scoped = TenantScopedCorpus(corpus, "tenant-a")
    report = scoped.ingest(md_book)

    deleted = scoped.remove(report.source_ids[0])

    assert deleted > 0
    assert corpus.count() == 0
