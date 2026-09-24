"""ClaudeContextWriter: sends short numbered labels (never real chunk_id
hashes) and correctly maps the model's per-index blurbs back onto the
original chunk order. No API calls: a stub client stands in for
anthropic.Anthropic.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from ragcore.contextual.claude import ClaudeContextWriter, _Blurb, _BlurbResult
from ragcore.types import Chunk, ChunkMeta


def _chunk(idx: int, chunk_id: str, text: str) -> Chunk:
    meta = ChunkMeta(
        source_id="src", source_path="book.epub", source_title="Book", source_type="epub",
        section_path=(), chapter_title=None, chapter_index=0, section=None,
        page_start=None, page_end=None, chunk_index=idx, context_group_id="src:group:0",
        char_start=0, char_end=len(text), token_count=len(text) // 4,
    )
    return Chunk(chunk_id=chunk_id, text=text, meta=meta)


class _StubResponse:
    def __init__(self, parsed_output: _BlurbResult) -> None:
        self.parsed_output = parsed_output


class _StubMessages:
    def __init__(self, result: _BlurbResult) -> None:
        self._result = result
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse(self._result)


class _StubClient:
    def __init__(self, result: _BlurbResult) -> None:
        self.messages = _StubMessages(result)


def test_sends_numeric_labels_never_real_chunk_ids() -> None:
    chunks = [_chunk(0, "hash:0", "alpha"), _chunk(1, "hash:1", "beta")]
    result = _BlurbResult(blurbs=[_Blurb(chunk_index=1, blurb="b1"), _Blurb(chunk_index=2, blurb="b2")])
    client = _StubClient(result)
    writer = ClaudeContextWriter(client=client)

    blurbs = writer.write_blurbs(chunks)

    assert blurbs == ["b1", "b2"]
    sent_content = client.messages.calls[0]["messages"][0]["content"]
    assert "[1]" in sent_content and "[2]" in sent_content
    assert "hash:0" not in sent_content and "hash:1" not in sent_content


def test_missing_index_degrades_to_empty_blurb_not_a_crash() -> None:
    chunks = [_chunk(0, "hash:0", "alpha"), _chunk(1, "hash:1", "beta")]
    result = _BlurbResult(blurbs=[_Blurb(chunk_index=1, blurb="b1")])  # index 2 missing
    client = _StubClient(result)
    writer = ClaudeContextWriter(client=client)

    blurbs = writer.write_blurbs(chunks)

    assert blurbs == ["b1", ""]


def test_empty_chunks_makes_no_api_call() -> None:
    client = _StubClient(_BlurbResult(blurbs=[]))
    writer = ClaudeContextWriter(client=client)

    assert writer.write_blurbs([]) == []
    assert client.messages.calls == []
