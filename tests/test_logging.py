"""Structured logging via the stdlib `logging` module
(docs/ARCHITECTURE.md Phase 6 gap list item 25: "zero logging calls found
anywhere in ragcore"). No handler/formatter is configured by the library
itself (see ragcore/corpus.py's module docstring for why) — these tests
use pytest's `caplog` fixture, which attaches its own handler, exactly
the way a consuming application would.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ragcore.corpus import Corpus
from tests.fakes import FakeEmbedder, InMemoryStore


def test_ingest_logs_an_info_summary_on_completion(caplog: pytest.LogCaptureFixture, md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    with caplog.at_level(logging.INFO, logger="ragcore.corpus"):
        corpus.ingest(md_book)

    assert any("ingest" in r.message and "done" in r.message for r in caplog.records)
    assert all(r.name == "ragcore.corpus" for r in caplog.records)


def test_ingest_debug_logs_are_silent_at_info_level(caplog, md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    with caplog.at_level(logging.INFO, logger="ragcore.corpus"):
        corpus.ingest(md_book)

    assert not any(r.levelno == logging.DEBUG for r in caplog.records)


def test_ingest_debug_logs_appear_at_debug_level(caplog, md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    with caplog.at_level(logging.DEBUG, logger="ragcore.corpus"):
        corpus.ingest(md_book)

    assert any(r.levelno == logging.DEBUG for r in caplog.records)
    assert any("chunked into" in r.message for r in caplog.records)


def test_retrieve_logs_at_debug_not_info(caplog, md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    caplog.clear()

    with caplog.at_level(logging.DEBUG, logger="ragcore.corpus"):
        corpus.retrieve("background on chunking", k=3)

    retrieve_records = [r for r in caplog.records if "retrieve:" in r.message]
    assert retrieve_records
    assert all(r.levelno == logging.DEBUG for r in retrieve_records)


def test_ingest_many_logs_a_retry_warning(caplog, md_book: Path) -> None:
    from ragcore.ingestion.sqlite_store import SqliteIngestCheckpointStore

    class FlakyOnce:
        """Fails the very first ingest() call, succeeds after."""

        def __init__(self, inner_embedder: FakeEmbedder) -> None:
            self._inner = inner_embedder
            self._failed_once = False
            self.name = inner_embedder.name
            self.dim = inner_embedder.dim

        def embed_groups(self, groups, *, input_type="document"):
            if not self._failed_once:
                self._failed_once = True
                raise RuntimeError("simulated transient failure")
            return self._inner.embed_groups(groups, input_type=input_type)

        def embed_query(self, text):
            return self._inner.embed_query(text)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_store = SqliteIngestCheckpointStore(f"{tmp}/checkpoints.sqlite3")
        corpus = Corpus(
            InMemoryStore(dim=8),
            FlakyOnce(FakeEmbedder(dim=8)),
            checkpoint_store=checkpoint_store,
        )
        with caplog.at_level(logging.WARNING, logger="ragcore.corpus"):
            report = corpus.ingest_many([md_book], max_retries=1)

    assert report.errors == []
    assert any("retrying" in r.message for r in caplog.records)
    assert all(r.levelno == logging.WARNING for r in caplog.records)
