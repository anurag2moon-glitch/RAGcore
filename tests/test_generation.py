"""Corpus.ask() composition tests, against FakeEmbedder + InMemoryStore +
FakeGenerator (no network, no anthropic/voyageai/lancedb dependency).
Proves ask() is exactly retrieve() + generate() composed, and that a Corpus
with no generator configured stays fully usable for retrieve() and only
fails ask() with a clear error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragcore.corpus import Corpus
from ragcore.errors import RagcoreError
from tests.fakes import FakeEmbedder, FakeGenerator, InMemoryStore


def test_ask_without_a_configured_generator_raises_clearly(md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))  # no generator
    corpus.ingest(md_book)

    with pytest.raises(RagcoreError, match="no Generator configured"):
        corpus.ask("what happens in chapter one")


def test_ask_composes_retrieve_and_generate(md_book: Path) -> None:
    generator = FakeGenerator()
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8), generator=generator)
    corpus.ingest(md_book)

    query = "background on chunking and embeddings"
    expected_ids = [r.chunk.chunk_id for r in corpus.retrieve(query, k=3)]

    answer = corpus.ask(query, k=3)

    assert generator.calls == [(query, expected_ids)]
    assert [r.chunk.chunk_id for r in answer.citations] == expected_ids
    assert query in answer.text


def test_ask_respects_filters_same_as_retrieve(md_book: Path) -> None:
    generator = FakeGenerator()
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8), generator=generator)
    corpus.ingest(md_book)

    filters = {"chapter_index": {"gte": 1}}
    query = "chunking"
    expected_ids = [r.chunk.chunk_id for r in corpus.retrieve(query, k=5, filters=filters)]

    answer = corpus.ask(query, k=5, filters=filters)

    assert [r.chunk.chunk_id for r in answer.citations] == expected_ids


def test_ask_accepts_a_per_call_generator_override(md_book: Path) -> None:
    default_generator = FakeGenerator()
    override_generator = FakeGenerator()
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8), generator=default_generator)
    corpus.ingest(md_book)

    corpus.ask("chunking", k=2, generator=override_generator)

    assert override_generator.calls
    assert default_generator.calls == []
