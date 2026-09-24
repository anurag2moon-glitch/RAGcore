"""build_contextual_embeddings() resumability: mirrors
test_concept_resume.py's guarantees exactly — a chapter that fails during
blurb-writing must leave no partial checkpoint, a re-run retries only the
unfinished chapters, and a model change invalidates every chapter's
checkpoint.
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


class _FlakyWriter:
    def __init__(self, fail_on_call: int) -> None:
        self._fail_on_call = fail_on_call
        self.calls: list[list[str]] = []
        self.model = "flaky"

    def write_blurbs(self, chunks: Sequence[Chunk]) -> list[str]:
        self.calls.append([c.chunk_id for c in chunks])
        if len(self.calls) == self._fail_on_call:
            raise RuntimeError(f"simulated failure on call {len(self.calls)}")
        return [f"blurb {len(self.calls)}-{i}" for i in range(len(chunks))]


def test_a_failed_chapter_leaves_no_partial_blurbs_or_checkpoint(md_book: Path) -> None:
    corpus = _build_corpus(md_book)
    store = SqliteContextStore(":memory:")
    writer = _FlakyWriter(fail_on_call=1)
    before_vectors = {c.chunk_id: c.vector for c in corpus.store.all_chunks()}

    report = build_contextual_embeddings(corpus, store, writer)

    assert len(report.errors) == 1
    failed_source_id, failed_group_id, _msg = report.errors[0]
    assert store.checkpoint(failed_source_id, failed_group_id) is None

    failed_chunks = [c for c in corpus.store.all_chunks() if c.meta.context_group_id == failed_group_id]
    for c in failed_chunks:
        assert store.blurb(c.chunk_id) is None
        assert c.vector == before_vectors[c.chunk_id]  # never re-embedded


def test_rerun_after_failure_retries_only_the_unfinished_chapter(md_book: Path) -> None:
    corpus = _build_corpus(md_book)
    store = SqliteContextStore(":memory:")
    writer = _FlakyWriter(fail_on_call=1)

    report1 = build_contextual_embeddings(corpus, store, writer)
    assert len(report1.errors) == 1
    calls_after_first_run = len(writer.calls)

    writer._fail_on_call = -1  # never fail again
    report2 = build_contextual_embeddings(corpus, store, writer)

    assert report2.chapters_processed == 1  # only the previously-failed chapter
    assert report2.chapters_skipped == report1.chapters_processed
    assert len(writer.calls) == calls_after_first_run + 1


def test_a_model_change_invalidates_every_checkpoint(md_book: Path) -> None:
    corpus = _build_corpus(md_book)
    store = SqliteContextStore(":memory:")

    v1 = _FlakyWriter(fail_on_call=-1)
    v1.model = "model-v1"
    build_contextual_embeddings(corpus, store, v1)
    assert len(v1.calls) > 0

    v2 = _FlakyWriter(fail_on_call=-1)
    v2.model = "model-v2"
    report = build_contextual_embeddings(corpus, store, v2)

    assert report.chapters_skipped == 0
    assert len(v2.calls) == len(v1.calls)
