"""Verifies ClaudeEntityExtractor cites short numbered indices to the model
(not full chunk_id hashes — a real chunk_id costs ~38 tokens per citation
vs. ~1 for a bare number, and a chapter's entities/relations cite many of
them) and correctly remaps those indices back to real chunk_ids afterward.
No API calls: a stub client stands in for anthropic.Anthropic, returning a
hand-built parsed response.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from ragcore.graph.claude_extractor import (
    ClaudeEntityExtractor,
    _ExtractedEntity,
    _ExtractedRelation,
    _ExtractionResult,
    _format_chunks,
)
from ragcore.types import Chunk, ChunkMeta


def _chunk(idx: int, chunk_id: str, text: str) -> Chunk:
    meta = ChunkMeta(
        source_id="src",
        source_path="book.epub",
        source_title="Book",
        source_type="epub",
        section_path=(),
        chapter_title=None,
        chapter_index=0,
        section=None,
        page_start=None,
        page_end=None,
        chunk_index=idx,
        context_group_id="src:group:0",
        char_start=0,
        char_end=len(text),
        token_count=len(text) // 4,
    )
    return Chunk(chunk_id=chunk_id, text=text, meta=meta)


class _StubResponse:
    def __init__(self, parsed_output: _ExtractionResult) -> None:
        self.parsed_output = parsed_output


class _StubMessages:
    def __init__(self, result: _ExtractionResult) -> None:
        self._result = result
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse(self._result)


class _StubClient:
    def __init__(self, result: _ExtractionResult) -> None:
        self.messages = _StubMessages(result)


def test_format_chunks_uses_short_numeric_labels_not_chunk_ids() -> None:
    chunks = [_chunk(0, "hash:0", "alpha"), _chunk(1, "hash:1", "beta")]

    text, index_to_chunk_id = _format_chunks(chunks)

    assert text == "[1]\nalpha\n\n[2]\nbeta"
    assert index_to_chunk_id == {1: "hash:0", 2: "hash:1"}
    assert "hash:0" not in text and "hash:1" not in text


def test_extract_sends_numeric_labels_and_remaps_response_to_real_chunk_ids() -> None:
    chunks = [
        _chunk(0, "hash:0", "Mr. Bingley arrived."),
        _chunk(1, "hash:1", "He danced with Jane."),
    ]
    result = _ExtractionResult(
        entities=[
            _ExtractedEntity(
                name="Mr. Bingley", entity_type="person", aliases=["Bingley"], chunk_indices=[1, 2]
            ),
            _ExtractedEntity(name="Jane", entity_type="person", chunk_indices=[2]),
        ],
        relations=[
            _ExtractedRelation(
                source_name="Mr. Bingley", target_name="Jane",
                relation_type="danced_with", chunk_indices=[2],
            ),
        ],
    )
    client = _StubClient(result)
    extractor = ClaudeEntityExtractor(client=client)

    entities, relations = extractor.extract(chunks)

    bingley = next(e for e in entities if e.name == "Mr. Bingley")
    jane = next(e for e in entities if e.name == "Jane")
    assert bingley.source_chunk_ids == ("hash:0", "hash:1")
    assert jane.source_chunk_ids == ("hash:1",)

    assert len(relations) == 1
    assert relations[0].source_chunk_ids == ("hash:1",)

    # the model was actually sent short numeric labels, never the real hashes
    sent_content = client.messages.calls[0]["messages"][0]["content"]
    assert "[1]" in sent_content and "[2]" in sent_content
    assert "hash:0" not in sent_content
    assert "hash:1" not in sent_content


def test_extract_drops_out_of_range_indices_without_crashing() -> None:
    chunks = [_chunk(0, "hash:0", "text")]
    result = _ExtractionResult(
        entities=[_ExtractedEntity(name="X", entity_type="person", chunk_indices=[1, 99])],
        relations=[],
    )
    client = _StubClient(result)
    extractor = ClaudeEntityExtractor(client=client)

    entities, _relations = extractor.extract(chunks)

    assert entities[0].source_chunk_ids == ("hash:0",)  # index 99 silently dropped
