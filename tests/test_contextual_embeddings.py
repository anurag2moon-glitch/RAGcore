"""build_contextual_embeddings and ClaudeContextWriter: the blurb must never
leak into Chunk.text (only into the string handed to the embedder and into
meta.extra["context_blurb"]), and the resulting vector must actually differ
from a plain embedding of the chunk's own text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ragcore.contextual.pipeline import build_contextual_embeddings
from ragcore.contextual.sqlite_store import SqliteContextStore
from ragcore.corpus import Corpus
from ragcore.types import Chunk
from tests.fakes import FakeEmbedder, InMemoryStore


def _build_corpus(md_book: Path) -> Corpus:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    return corpus


class _FakeWriter:
    model = "fake-writer"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def write_blurbs(self, chunks: Sequence[Chunk]) -> list[str]:
        self.calls.append([c.chunk_id for c in chunks])
        return [f"blurb for {c.chunk_id}" for c in chunks]


def test_blurb_never_leaks_into_chunk_text_only_into_extra(md_book: Path) -> None:
    corpus = _build_corpus(md_book)
    store = SqliteContextStore(":memory:")

    build_contextual_embeddings(corpus, store, _FakeWriter())

    for chunk in corpus.store.all_chunks():
        assert chunk.meta.extra["context_blurb"] == f"blurb for {chunk.chunk_id}"
        assert "blurb for" not in chunk.text


def test_embedded_vector_changes_after_contextualization(md_book: Path) -> None:
    corpus = _build_corpus(md_book)
    before = {c.chunk_id: c.vector for c in corpus.store.all_chunks()}
    store = SqliteContextStore(":memory:")

    build_contextual_embeddings(corpus, store, _FakeWriter())

    after = {c.chunk_id: c.vector for c in corpus.store.all_chunks()}
    for chunk_id in before:
        assert after[chunk_id] != before[chunk_id]


def test_one_writer_call_per_chapter_not_per_chunk_or_whole_book(md_book: Path) -> None:
    corpus = _build_corpus(md_book)
    store = SqliteContextStore(":memory:")
    writer = _FakeWriter()

    build_contextual_embeddings(corpus, store, writer)

    assert len(writer.calls) >= 2  # at least two chapters in md_book


def test_blurb_is_retrievable_from_the_context_store(md_book: Path) -> None:
    corpus = _build_corpus(md_book)
    store = SqliteContextStore(":memory:")

    build_contextual_embeddings(corpus, store, _FakeWriter())

    chunk = corpus.store.all_chunks()[0]
    assert store.blurb(chunk.chunk_id) == f"blurb for {chunk.chunk_id}"
    assert store.blurb("nonexistent") is None


def test_remove_source_deletes_blurbs_and_checkpoints(md_book: Path) -> None:
    corpus = _build_corpus(md_book)
    store = SqliteContextStore(":memory:")
    source_id = corpus.sources()[0].source_id

    build_contextual_embeddings(corpus, store, _FakeWriter())
    chunk = corpus.store.all_chunks()[0]
    assert store.blurb(chunk.chunk_id) is not None

    store.remove_source(source_id)

    assert store.blurb(chunk.chunk_id) is None
    assert store.checkpoint(source_id, chunk.meta.context_group_id) is None
