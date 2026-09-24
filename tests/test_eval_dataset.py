"""ragcore.eval.dataset: JSONL round-trip and the find_gold_passage
authoring helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragcore.corpus import Corpus
from ragcore.eval.dataset import find_gold_passage, load_gold_set, save_gold_set
from ragcore.eval.types import GoldPassage, GoldQuestion
from tests.fakes import FakeEmbedder, InMemoryStore


def test_save_and_load_gold_set_round_trips(tmp_path: Path) -> None:
    questions = [
        GoldQuestion(
            question_id="q1",
            query="what happened",
            gold_passages=(
                GoldPassage(source_path="book.md", chapter_index=1),
                GoldPassage(chunk_ids=("h:0", "h:1")),
            ),
            expected_facts=("fact one",),
            notes="a note",
        ),
        GoldQuestion(question_id="q2", query="another question", gold_passages=()),
    ]
    path = tmp_path / "gold.jsonl"

    save_gold_set(path, questions)
    loaded = load_gold_set(path)

    assert loaded == list(questions)
    # confirm it's really JSONL — one JSON object per line
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2


def test_find_gold_passage_resolves_a_marker_to_a_real_chunk(md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    source_title = corpus.sources()[0].title

    passage = find_gold_passage(corpus, source_title, "contextualized retrieval")

    assert passage.chunk_ids
    hydrated = corpus.get(list(passage.chunk_ids))
    assert "contextualized retrieval" in hydrated[0].text
    assert passage.source_id == corpus.sources()[0].source_id


def test_find_gold_passage_raises_when_marker_not_found(md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    source_title = corpus.sources()[0].title

    with pytest.raises(ValueError, match="no chunk containing"):
        find_gold_passage(corpus, source_title, "this text does not exist anywhere")
