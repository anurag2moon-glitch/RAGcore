"""TenantScopedCorpus: enforced tenant_id on every ingest/retrieve/ask/
get/sources/remove/count call, not just made possible by a filter the
caller has to remember (docs/adr/0006-tenant-id-foundation-not-enforcement.md).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragcore.corpus import Corpus
from ragcore.errors import TenancyError
from ragcore.graph.sqlite_store import SqliteGraphStore
from ragcore.graph.types import Entity
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


def test_generator_property_delegates_to_the_wrapped_corpus() -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    scoped = TenantScopedCorpus(corpus, "tenant-a")
    assert scoped.generator is None

    from tests.fakes import FakeGenerator

    corpus.generator = FakeGenerator()
    assert scoped.generator is corpus.generator


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


def test_hard_delete_removes_this_tenants_vector_chunks_only(md_book: Path, tmp_path: Path) -> None:
    corpus, book_a, book_b = _two_tenant_corpus(md_book, tmp_path)
    scoped_a = TenantScopedCorpus(corpus, "tenant-a")
    scoped_b = TenantScopedCorpus(corpus, "tenant-b")
    scoped_a.ingest(book_a)
    scoped_b.ingest(book_b)

    report = scoped_a.hard_delete()

    assert report.tenant_id == "tenant-a"
    assert report.chunks_deleted > 0
    assert corpus.count() == scoped_b.count()
    assert all(c.meta.tenant_id == "tenant-b" for c in corpus.store.all_chunks())


def test_hard_delete_is_safe_against_a_tier_2_shared_entity_id(
    md_book: Path, tmp_path: Path
) -> None:
    """Simulates the exact danger docs/adr/0007 names: two tenants whose
    extraction both produce the same content-addressed entity_id (e.g.
    two "John Smith" entities, normalized identically). hard_delete for
    one tenant must remove only *its own* chunk_id's association with
    that shared row, leaving the other tenant's association and the row
    itself (since it's still referenced) untouched."""
    corpus, book_a, book_b = _two_tenant_corpus(md_book, tmp_path)
    scoped_a = TenantScopedCorpus(corpus, "tenant-a")
    scoped_b = TenantScopedCorpus(corpus, "tenant-b")
    report_a = scoped_a.ingest(book_a)
    report_b = scoped_b.ingest(book_b)

    chunk_a = next(c.chunk_id for c in corpus.store.get_by_source(report_a.source_ids[0]))
    chunk_b = next(c.chunk_id for c in corpus.store.get_by_source(report_b.source_ids[0]))

    graph_store = SqliteGraphStore(tmp_path / "graph.sqlite")
    shared_id = "entity:person:john-smith"
    graph_store.add_entities([Entity(entity_id=shared_id, name="John Smith", entity_type="person", source_chunk_ids=(chunk_a,))])
    graph_store.add_entities([Entity(entity_id=shared_id, name="John Smith", entity_type="person", source_chunk_ids=(chunk_b,))])
    [merged] = graph_store.entities_by_type("person")
    assert set(merged.source_chunk_ids) == {chunk_a, chunk_b}

    scoped_a.hard_delete(graph_store=graph_store)

    remaining = graph_store.entities_by_type("person")
    assert len(remaining) == 1
    assert remaining[0].entity_id == shared_id
    assert remaining[0].source_chunk_ids == (chunk_b,)  # tenant-a's association gone, tenant-b's intact


def test_hard_delete_wipes_an_auxiliary_lexical_store_scoped_to_this_tenant(
    md_book: Path, tmp_path: Path
) -> None:
    from ragcore.lexical.fts5 import Fts5LexicalStore

    corpus, book_a, book_b = _two_tenant_corpus(md_book, tmp_path)
    scoped_a = TenantScopedCorpus(corpus, "tenant-a")
    scoped_b = TenantScopedCorpus(corpus, "tenant-b")
    report_a = scoped_a.ingest(book_a)
    report_b = scoped_b.ingest(book_b)

    lexical = Fts5LexicalStore(tmp_path / "lexical.sqlite")
    lexical.index(corpus.store.get_by_source(report_a.source_ids[0]))
    lexical.index(corpus.store.get_by_source(report_b.source_ids[0]))
    assert lexical.count() > 0
    count_before = lexical.count()

    scoped_a.hard_delete(lexical_store=lexical)

    assert 0 < lexical.count() < count_before
