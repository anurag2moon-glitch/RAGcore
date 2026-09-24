"""build_concepts() resumability: a chapter that fails mid-extraction must
leave no partial concepts/edges behind (the single-transaction guarantee in
SqliteConceptStore.add_chapter_concepts, not just the checkpoint row), and a
re-run must retry only the chapters that never completed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ragcore.corpus import Corpus
from ragcore.concepts.pipeline import build_concepts
from ragcore.concepts.sqlite_store import SqliteConceptStore
from ragcore.concepts.types import Concept
from ragcore.types import Chunk
from tests.fakes import FakeEmbedder, InMemoryStore


def _build_corpus(md_book: Path) -> Corpus:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    return corpus


class _FlakyExtractor:
    """Raises on the Nth call (1-indexed), succeeds otherwise — simulates a
    mid-corpus API failure without any network involved."""

    model = "flaky"

    def __init__(self, fail_on_call: int) -> None:
        self._fail_on_call = fail_on_call
        self.calls: list[list[str]] = []

    def extract(self, chunks: Sequence[Chunk]) -> tuple[list[Concept], list]:
        self.calls.append([c.chunk_id for c in chunks])
        if len(self.calls) == self._fail_on_call:
            raise RuntimeError(f"simulated failure on call {len(self.calls)}")
        concept_id = f"concept:c{len(self.calls)}"
        return (
            [Concept(concept_id=concept_id, label=f"idea {len(self.calls)}", description="d",
                     source_chunk_ids=tuple(c.chunk_id for c in chunks))],
            [],
        )


def test_a_failed_chapter_leaves_no_partial_concepts_or_checkpoint(md_book: Path) -> None:
    corpus = _build_corpus(md_book)
    store = SqliteConceptStore(":memory:")
    extractor = _FlakyExtractor(fail_on_call=1)

    report = build_concepts(corpus, store, extractor)

    assert len(report.errors) == 1
    failed_source_id, failed_group_id, _msg = report.errors[0]
    assert store.checkpoint(failed_source_id, failed_group_id) is None
    # No concept from the failed chapter's own extraction should exist —
    # concept:c1 was the id the (never-returned) result would have used.
    assert store.concept("concept:c1") is None


def test_rerun_after_failure_retries_only_the_unfinished_chapter(md_book: Path) -> None:
    corpus = _build_corpus(md_book)
    store = SqliteConceptStore(":memory:")
    extractor = _FlakyExtractor(fail_on_call=1)

    report1 = build_concepts(corpus, store, extractor)
    assert report1.chapters_processed >= 1  # later chapters still succeeded
    assert len(report1.errors) == 1
    calls_after_first_run = len(extractor.calls)

    # Second run: the chapter that failed has no matching checkpoint and
    # must be retried; every chapter that already succeeded must not be
    # re-extracted (its checkpoint fingerprint already matches).
    extractor._fail_on_call = -1  # never fail again
    report2 = build_concepts(corpus, store, extractor)

    assert report2.chapters_processed == 1  # only the one previously-failed chapter
    assert report2.chapters_skipped == report1.chapters_processed
    assert len(extractor.calls) == calls_after_first_run + 1


def test_resume_false_reprocesses_every_chapter_regardless_of_checkpoints(md_book: Path) -> None:
    corpus = _build_corpus(md_book)
    store = SqliteConceptStore(":memory:")
    extractor = _FlakyExtractor(fail_on_call=-1)

    build_concepts(corpus, store, extractor)
    calls_after_first = len(extractor.calls)

    report2 = build_concepts(corpus, store, extractor, resume=False)

    assert report2.chapters_skipped == 0
    assert len(extractor.calls) == calls_after_first * 2


def test_a_model_change_invalidates_the_checkpoint_and_forces_reextraction(md_book: Path) -> None:
    corpus = _build_corpus(md_book)
    store = SqliteConceptStore(":memory:")

    class _NamedExtractor(_FlakyExtractor):
        def __init__(self, model: str) -> None:
            super().__init__(fail_on_call=-1)
            self.model = model

    v1 = _NamedExtractor("model-v1")
    build_concepts(corpus, store, v1)
    assert len(v1.calls) > 0

    v2 = _NamedExtractor("model-v2")
    report = build_concepts(corpus, store, v2)

    assert report.chapters_skipped == 0  # different model -> different fingerprint -> no skip
    assert len(v2.calls) == len(v1.calls)
