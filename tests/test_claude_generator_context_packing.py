"""ClaudeGenerator's opt-in context_max_tokens path (pack_context wired
in) — no API calls, a stub client stands in for anthropic.Anthropic,
matching test_claude_sufficiency_judge.py's pattern but for
messages.create (ClaudeGenerator's actual call) rather than
messages.parse.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from ragcore.chunking.tokenizer import count_tokens
from ragcore.generation.claude import ClaudeGenerator
from ragcore.types import Chunk, ChunkMeta, Retrieved


def _chunk(chunk_id: str, text: str, chunk_index: int = 0) -> Chunk:
    meta = ChunkMeta(
        source_id="src", source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=0, section=None,
        page_start=None, page_end=None, chunk_index=chunk_index, context_group_id="g",
        char_start=chunk_index * 100, char_end=chunk_index * 100 + len(text), token_count=1,
    )
    return Chunk(chunk_id=chunk_id, text=text, meta=meta)


class _TextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _StubResponse:
    stop_reason = "end_turn"
    stop_details = None

    def __init__(self, text: str = "an answer") -> None:
        self.content = [_TextBlock(text)]


class _StubMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse()


class _StubClient:
    def __init__(self) -> None:
        self.messages = _StubMessages()


def test_without_context_max_tokens_every_chunk_is_sent_and_cited() -> None:
    client = _StubClient()
    generator = ClaudeGenerator(client=client)
    big_text = "word " * 50
    context = [Retrieved(chunk=_chunk("c1", big_text, 0), score=0.9), Retrieved(chunk=_chunk("c2", big_text, 1), score=0.1)]

    answer = generator.generate("q", context)

    sent_content = client.messages.calls[0]["messages"][0]["content"]
    assert "c1" not in sent_content  # chunk_id itself never appears in the prompt text
    assert sent_content.count("word") == 100  # both chunks' full text present
    assert len(answer.citations) == 2


def test_context_max_tokens_drops_what_does_not_fit() -> None:
    client = _StubClient()
    big_text = "word " * 50
    tokens_per_chunk = count_tokens(big_text)
    generator = ClaudeGenerator(client=client, context_max_tokens=tokens_per_chunk + 5)
    context = [
        Retrieved(chunk=_chunk("c1", big_text, 0), score=0.9),
        Retrieved(chunk=_chunk("c2", big_text, 1), score=0.1),
    ]

    answer = generator.generate("q", context)

    assert len(answer.citations) == 1
    assert answer.citations[0].chunk.chunk_id == "c1"
    sent_content = client.messages.calls[0]["messages"][0]["content"]
    assert sent_content.count("word") == 50  # only one chunk's text made the prompt


def test_context_max_tokens_zero_sends_no_context_chunks() -> None:
    client = _StubClient()
    generator = ClaudeGenerator(client=client, context_max_tokens=0)
    context = [Retrieved(chunk=_chunk("c1", "some text"), score=0.9)]

    answer = generator.generate("q", context)

    assert answer.citations == []
