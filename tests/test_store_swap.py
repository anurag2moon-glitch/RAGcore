"""Swap test: the same corpus fixture through two VectorStore
implementations (InMemoryStore and LanceDBStore) should retrieve chunks in
identical order. Catches store-specific logic leaking into Corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragcore.corpus import Corpus
from tests.fakes import FakeEmbedder, InMemoryStore

lancedb = pytest.importorskip("lancedb")


def test_store_swap_preserves_retrieval_ordering(md_book: Path, tmp_path: Path) -> None:
    from ragcore.stores.lancedb import LanceDBStore

    embedder = FakeEmbedder(dim=8)

    mem_corpus = Corpus(InMemoryStore(dim=8), embedder)
    mem_corpus.ingest(md_book)

    lance_corpus = Corpus(LanceDBStore(tmp_path / "lancedb", dim=8), embedder)
    lance_corpus.ingest(md_book)

    query = "background on chunking and embeddings"
    mem_results = mem_corpus.retrieve(query, k=5)
    lance_results = lance_corpus.retrieve(query, k=5)

    assert [r.chunk.chunk_id for r in mem_results] == [r.chunk.chunk_id for r in lance_results]
    # metadata round-trips through LanceDB's columnar schema too
    assert mem_results[0].chunk.meta.section_path == lance_results[0].chunk.meta.section_path

    mem_sources = {s.source_id: s for s in mem_corpus.sources()}
    lance_sources = {s.source_id: s for s in lance_corpus.sources()}
    assert mem_sources.keys() == lance_sources.keys()
    for sid, mem_info in mem_sources.items():
        lance_info = lance_sources[sid]
        assert lance_info.chunk_count == mem_info.chunk_count
        assert lance_info.path == mem_info.path
        assert lance_info.title == mem_info.title
        assert lance_info.source_type == mem_info.source_type
