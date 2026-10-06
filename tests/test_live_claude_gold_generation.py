"""Live smoke test against the real Claude API. Excluded by default (see
pyproject.toml `addopts = "-m 'not live'"`); run explicitly with
`pytest -m live`. Requires ANTHROPIC_API_KEY and VOYAGE_API_KEY in the
environment (ingestion needs a real embedder too).
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


@pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("VOYAGE_API_KEY")),
    reason="ANTHROPIC_API_KEY/VOYAGE_API_KEY not set",
)
def test_claude_gold_set_generator_produces_a_usable_gold_set(tmp_path) -> None:
    from ragcore.corpus import Corpus
    from ragcore.eval.gold_generation import ClaudeGoldSetGenerator
    from ragcore.eval.report import run_eval

    book = tmp_path / "book.md"
    book.write_text(
        "# Chapter 1: The Mitochondria\n\n"
        "The mitochondria is the powerhouse of the cell. It produces ATP "
        "through cellular respiration, converting glucose and oxygen into "
        "usable chemical energy.\n"
    )

    corpus = Corpus.open(tmp_path / "live_gold_gen.lancedb")
    corpus.ingest(book)

    questions = ClaudeGoldSetGenerator(questions_per_chapter=1).generate(corpus)

    assert len(questions) >= 1
    [question] = questions[:1]
    assert question.query
    assert question.gold_passages[0].chunk_ids

    report = run_eval(corpus, questions, k=5)
    assert report.retrieval.recall_at_k > 0
