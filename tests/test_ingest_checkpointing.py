"""Corpus-level integration tests for opt-in mid-ingest resumability
(checkpoint_store=...) and ingest_many()'s dead-letter handling — see
docs/adr/0002-ingest-checkpointing-opt-in.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import pytest

from ragcore.corpus import Corpus
from ragcore.ingestion.sqlite_store import SqliteIngestCheckpointStore
from ragcore.types import Vector
from tests.fakes import FakeEmbedder, InMemoryStore


class FlakyEmbedder:
    """Wraps a real Embedder double and raises on exactly the Nth call to
    embed_groups() (1-indexed, across this instance's whole lifetime) —
    simulates a transient failure partway through a multi-chapter
    ingest(), since Corpus._ingest_checkpointed() embeds one ChunkGroup
    (one call to embed_groups) at a time."""

    def __init__(self, base: FakeEmbedder, fail_at_call: int) -> None:
        self._base = base
        self._fail_at_call = fail_at_call
        self._calls = 0
        self.embed_calls: list[list[str]] = []
        self.dim = base.dim
        self.name = base.name

    def embed_groups(
        self,
        groups: Sequence[Sequence[str]],
        *,
        input_type: Literal["document", "query", None] = "document",
    ) -> list[list[Vector]]:
        self._calls += 1
        for group in groups:
            self.embed_calls.append(list(group))
        if self._calls == self._fail_at_call:
            raise RuntimeError("simulated embedding failure")
        return self._base.embed_groups(groups, input_type=input_type)

    def embed_query(self, text: str) -> Vector:
        return self._base.embed_query(text)


def test_resumes_after_mid_ingest_failure(tmp_path: Path, md_book: Path) -> None:
    base = FakeEmbedder(dim=8)
    embedder = FlakyEmbedder(base, fail_at_call=2)  # chapter 2's embed call fails
    store = InMemoryStore(dim=8)
    checkpoint_store = SqliteIngestCheckpointStore(tmp_path / "checkpoints.sqlite3")
    corpus = Corpus(store, embedder, checkpoint_store=checkpoint_store)

    with pytest.raises(RuntimeError, match="simulated embedding failure"):
        corpus.ingest(md_book)

    # Chapter 1's chunks landed and were checkpointed; chapter 2's did not.
    assert store.count() == 2
    [source] = store.list_sources()
    source_id = source.source_id
    assert checkpoint_store.checkpoint(source_id, f"{source_id}:group:0") is not None
    assert checkpoint_store.checkpoint(source_id, f"{source_id}:group:1") is None
    chapter1_calls_before_retry = len(embedder.embed_calls)

    report = corpus.ingest(md_book)

    assert report.chunks_ingested == 3
    assert store.count() == 3
    # Chapter 1 was not re-embedded: the only new embed_groups call on
    # retry is chapter 2's.
    assert len(embedder.embed_calls) == chapter1_calls_before_retry + 1
    assert checkpoint_store.checkpoint(source_id, f"{source_id}:group:1") is not None


def test_replace_does_not_delete_already_resumed_chunks(tmp_path: Path, md_book: Path) -> None:
    base = FakeEmbedder(dim=8)
    embedder = FlakyEmbedder(base, fail_at_call=2)
    store = InMemoryStore(dim=8)
    checkpoint_store = SqliteIngestCheckpointStore(tmp_path / "checkpoints.sqlite3")
    corpus = Corpus(store, embedder, checkpoint_store=checkpoint_store)

    with pytest.raises(RuntimeError):
        corpus.ingest(md_book, replace=True)

    [source] = store.list_sources()
    chunk0_before = store.get([f"{source.source_id}:0"])[0]
    vector_before = chunk0_before.vector

    corpus.ingest(md_book, replace=True)

    chunk0_after = store.get([chunk0_before.chunk_id])[0]
    assert chunk0_after.vector == vector_before  # never re-embedded or deleted+reinserted


def test_ingest_without_checkpoint_store_has_no_resumability(tmp_path: Path, md_book: Path) -> None:
    """Without checkpoint_store, a failure anywhere aborts the whole
    ingest() with nothing stored — today's pre-existing behavior,
    confirmed unchanged. Unlike the checkpointed path, the single-shot
    path embeds every ChunkGroup in one embed_groups() call, so
    fail_at_call=1 (not 2) is what exercises "fails partway through"
    here."""
    base = FakeEmbedder(dim=8)
    embedder = FlakyEmbedder(base, fail_at_call=1)
    store = InMemoryStore(dim=8)
    corpus = Corpus(store, embedder)  # no checkpoint_store

    with pytest.raises(RuntimeError):
        corpus.ingest(md_book)

    assert store.count() == 0  # single-shot: nothing upserted until the end


class AlwaysFailingEmbedder:
    """Fails every embed_groups() call unconditionally — a permanently
    broken file/config, as opposed to FlakyEmbedder's one-shot transient
    failure."""

    name = "always-failing-test-embedder"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.embed_calls: list[list[str]] = []

    def embed_groups(self, groups: Sequence[Sequence[str]], *, input_type="document"):
        for group in groups:
            self.embed_calls.append(list(group))
        raise RuntimeError("permanent failure")

    def embed_query(self, text: str) -> Vector:
        raise RuntimeError("permanent failure")


def test_ingest_many_dead_letters_after_exhausting_retries(tmp_path: Path, md_book: Path) -> None:
    embedder = AlwaysFailingEmbedder(dim=8)
    store = InMemoryStore(dim=8)
    checkpoint_store = SqliteIngestCheckpointStore(tmp_path / "checkpoints.sqlite3")
    corpus = Corpus(store, embedder, checkpoint_store=checkpoint_store)

    report = corpus.ingest_many([md_book], max_retries=2)  # 3 total attempts

    assert report.errors == [(str(md_book), "permanent failure")]
    [dl] = checkpoint_store.dead_letters()
    assert dl.path == str(md_book)
    assert dl.attempts == 3
    assert dl.error == "permanent failure"


def test_ingest_many_retries_then_succeeds_without_restarting_from_scratch(
    tmp_path: Path, md_book: Path
) -> None:
    base = FakeEmbedder(dim=8)
    # Fails on the 2nd embed_groups call (chapter 2's first attempt), then
    # every later call succeeds — so the retry must resume past chapter 1.
    embedder = FlakyEmbedder(base, fail_at_call=2)
    store = InMemoryStore(dim=8)
    checkpoint_store = SqliteIngestCheckpointStore(tmp_path / "checkpoints.sqlite3")
    corpus = Corpus(store, embedder, checkpoint_store=checkpoint_store)

    report = corpus.ingest_many([md_book], max_retries=1)

    assert report.errors == []
    assert report.chunks_ingested == 3
    assert checkpoint_store.dead_letters() == []
    # Chapter 1 embedded exactly once across both attempts.
    chapter1_embeds = [c for c in embedder.embed_calls if len(c) == 2]
    assert len(chapter1_embeds) == 1


def test_ingest_many_clears_a_previously_dead_lettered_path_on_success(
    tmp_path: Path, md_book: Path
) -> None:
    base = FakeEmbedder(dim=8)
    store = InMemoryStore(dim=8)
    checkpoint_store = SqliteIngestCheckpointStore(tmp_path / "checkpoints.sqlite3")
    checkpoint_store.mark_dead_letter(path=str(md_book), error="old failure", attempts=5)
    corpus = Corpus(store, base, checkpoint_store=checkpoint_store)

    report = corpus.ingest_many([md_book])

    assert report.errors == []
    assert checkpoint_store.dead_letters() == []
