"""Live smoke test against the real Claude API. Excluded by default (see
pyproject.toml `addopts = "-m 'not live'"`); run explicitly with
`pytest -m live`. Requires ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import os

import pytest

from ragcore.types import Chunk, ChunkMeta, Retrieved

pytestmark = pytest.mark.live


def _fake_context() -> list[Retrieved]:
    meta = ChunkMeta(
        source_id="src1",
        source_path="book.epub",
        source_title="Test Book",
        source_type="epub",
        section_path=("Chapter I.",),
        chapter="Chapter I.",
        chapter_index=0,
        section=None,
        page_start=1,
        page_end=1,
        chunk_index=0,
        context_group_id="src1:group:0",
        char_start=0,
        char_end=42,
        token_count=10,
    )
    chunk = Chunk(
        chunk_id="src1:0",
        text="Sir William Lucas was formerly in trade in Meryton, where he "
        "made a tolerable fortune, and was knighted for an address to the "
        "king during his mayoralty.",
        meta=meta,
    )
    return [Retrieved(chunk=chunk, score=0.9)]


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_claude_generator_answers_grounded_in_context() -> None:
    from ragcore.generation.claude import ClaudeGenerator

    generator = ClaudeGenerator()
    answer = generator.generate("Who is Sir William Lucas?", _fake_context())

    assert answer.text
    assert "Lucas" in answer.text
    assert answer.citations[0].chunk.chunk_id == "src1:0"
