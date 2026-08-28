"""Chunk id stability across re-ingestion — the mechanism the entire
future graph upgrade depends on (see design doc §4: graph edges are keyed
on chunk_id and must stay valid after a book is re-ingested).
"""

from __future__ import annotations

from pathlib import Path

from ragcore.corpus import Corpus
from tests.fakes import FakeEmbedder, InMemoryStore


def test_chunk_ids_stable_across_reingestion(md_book: Path) -> None:
    store = InMemoryStore(dim=8)
    corpus = Corpus(store, FakeEmbedder(dim=8))

    report1 = corpus.ingest(md_book)
    ids1 = {c.chunk_id for c in store.all_chunks()}
    assert ids1

    report2 = corpus.ingest(md_book)  # replace=True by default
    ids2 = {c.chunk_id for c in store.all_chunks()}

    assert ids1 == ids2
    assert report2.sources_replaced == report1.source_ids
    assert corpus.count() == report2.chunks_ingested  # no duplicate rows left behind


def test_chunk_ids_are_content_addressed(md_book: Path, tmp_path: Path) -> None:
    """The same content ingested from two different files (different paths)
    still gets the same source_id / chunk_ids, since source_id hashes file
    bytes, not the path."""
    store = InMemoryStore(dim=8)
    corpus = Corpus(store, FakeEmbedder(dim=8))

    copy_path = tmp_path / "copy" / "book.md"
    copy_path.parent.mkdir()
    copy_path.write_bytes(md_book.read_bytes())

    corpus.ingest(md_book, replace=False)
    ids_from_original = {c.chunk_id for c in store.all_chunks()}

    corpus.ingest(copy_path, replace=False)
    ids_after_copy = {c.chunk_id for c in store.all_chunks()}

    # the copy's chunk_ids are identical to the original's, so upserting it
    # overwrote the same rows rather than appending new ones
    assert ids_after_copy == ids_from_original
