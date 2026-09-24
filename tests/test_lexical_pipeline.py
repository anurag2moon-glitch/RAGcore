"""build_lexical_index — the separate pass that populates a LexicalStore
from already-ingested chunks. No checkpoint table needed (see the module
docstring): indexing is idempotent and LLM-free, so a plain full-rebuild
pass is always correct and cheap.
"""

from __future__ import annotations

from pathlib import Path

from ragcore.corpus import Corpus
from ragcore.lexical.fts5 import Fts5LexicalStore
from ragcore.lexical.pipeline import build_lexical_index
from tests.fakes import FakeEmbedder, InMemoryStore


def _corpus_with_chunks(md_book: Path) -> Corpus:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    return corpus


def test_indexes_every_chunk_across_every_source(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    lexical = Fts5LexicalStore(":memory:")

    report = build_lexical_index(corpus, lexical)

    all_chunks = corpus.store.all_chunks()
    assert report.chunks_indexed == len(all_chunks)
    assert lexical.count() == len(all_chunks)


def test_source_ids_filter_restricts_which_sources_are_indexed(md_book: Path, txt_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    corpus.ingest(txt_book)
    lexical = Fts5LexicalStore(":memory:")
    wanted_source = corpus.sources()[0].source_id

    report = build_lexical_index(corpus, lexical, source_ids=[wanted_source])

    assert report.sources_indexed == 1
    expected = len(corpus.store.get_by_source(wanted_source))
    assert lexical.count() == expected


def test_rerun_is_idempotent(md_book: Path) -> None:
    corpus = _corpus_with_chunks(md_book)
    lexical = Fts5LexicalStore(":memory:")

    build_lexical_index(corpus, lexical)
    build_lexical_index(corpus, lexical)

    assert lexical.count() == len(corpus.store.all_chunks())
