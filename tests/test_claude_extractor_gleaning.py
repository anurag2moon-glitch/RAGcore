"""ClaudeEntityExtractor's opt-in gleaning_rounds (docs/ARCHITECTURE.md
Phase 4 gap list item 17) — no API calls, a stub client returns a
scripted sequence of responses, one per call (initial pass, then one per
gleaning round).
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from ragcore.graph.claude_extractor import (
    ClaudeEntityExtractor,
    _ExtractedEntity,
    _ExtractedRelation,
    _ExtractionResult,
)
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
    def __init__(self, parsed_output: _ExtractionResult) -> None:
        self.parsed_output = parsed_output


class _ScriptedMessages:
    """Returns one scripted _ExtractionResult per call, in order."""

    def __init__(self, results: list[_ExtractionResult]) -> None:
        self._results = results
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse(self._results[len(self.calls) - 1])


class _StubClient:
    def __init__(self, results: list[_ExtractionResult]) -> None:
        self.messages = _ScriptedMessages(results)


def test_zero_gleaning_rounds_makes_exactly_one_call() -> None:
    chunks = [_chunk(0, "hash:0", "Mr. Bingley arrived.")]
    result = _ExtractionResult(
        entities=[_ExtractedEntity(name="Mr. Bingley", entity_type="character", chunk_indices=[1])],
        relations=[],
    )
    client = _StubClient([result])
    extractor = ClaudeEntityExtractor(client=client)  # gleaning_rounds=0 default

    entities, _relations = extractor.extract(chunks)

    assert len(client.messages.calls) == 1
    assert [e.name for e in entities] == ["Mr. Bingley"]


def test_gleaning_round_adds_newly_found_entities() -> None:
    chunks = [_chunk(0, "hash:0", "Mr. Bingley arrived. Jane smiled.")]
    first = _ExtractionResult(
        entities=[_ExtractedEntity(name="Mr. Bingley", entity_type="character", chunk_indices=[1])],
        relations=[],
    )
    gleaned = _ExtractionResult(
        found_more=True,
        entities=[_ExtractedEntity(name="Jane", entity_type="character", chunk_indices=[1])],
        relations=[],
    )
    client = _StubClient([first, gleaned])
    extractor = ClaudeEntityExtractor(client=client, gleaning_rounds=1)

    entities, _relations = extractor.extract(chunks)

    assert len(client.messages.calls) == 2
    assert {e.name for e in entities} == {"Mr. Bingley", "Jane"}


def test_gleaning_stops_early_when_model_reports_nothing_more() -> None:
    chunks = [_chunk(0, "hash:0", "Mr. Bingley arrived.")]
    first = _ExtractionResult(
        entities=[_ExtractedEntity(name="Mr. Bingley", entity_type="character", chunk_indices=[1])],
        relations=[],
    )
    nothing_more = _ExtractionResult(found_more=False, entities=[], relations=[])
    # a third response would prove a bug if it were ever requested
    client = _StubClient([first, nothing_more, nothing_more])
    extractor = ClaudeEntityExtractor(client=client, gleaning_rounds=3)

    entities, _relations = extractor.extract(chunks)

    assert len(client.messages.calls) == 2  # initial + one gleaning round, then stopped
    assert [e.name for e in entities] == ["Mr. Bingley"]


def test_gleaning_round_runs_up_to_the_configured_limit() -> None:
    chunks = [_chunk(0, "hash:0", "text")]
    first = _ExtractionResult(entities=[], relations=[])
    always_finds_more = _ExtractionResult(
        found_more=True,
        entities=[_ExtractedEntity(name="X", entity_type="character", chunk_indices=[1])],
        relations=[],
    )
    client = _StubClient([first, always_finds_more, always_finds_more])
    extractor = ClaudeEntityExtractor(client=client, gleaning_rounds=2)

    extractor.extract(chunks)

    assert len(client.messages.calls) == 3  # initial + 2 gleaning rounds, limit respected


def test_gleaning_round_merges_chunk_ids_for_an_entity_found_again() -> None:
    chunks = [_chunk(0, "hash:0", "a"), _chunk(1, "hash:1", "b")]
    first = _ExtractionResult(
        entities=[_ExtractedEntity(name="Mr. Bingley", entity_type="character", chunk_indices=[1])],
        relations=[],
    )
    gleaned = _ExtractionResult(
        found_more=True,
        entities=[_ExtractedEntity(name="Mr. Bingley", entity_type="character", chunk_indices=[2])],
        relations=[],
    )
    client = _StubClient([first, gleaned])
    extractor = ClaudeEntityExtractor(client=client, gleaning_rounds=1)

    entities, _relations = extractor.extract(chunks)

    [bingley] = entities
    assert set(bingley.source_chunk_ids) == {"hash:0", "hash:1"}


def test_gleaning_conversation_includes_a_recap_of_prior_findings() -> None:
    chunks = [_chunk(0, "hash:0", "Mr. Bingley arrived.")]
    first = _ExtractionResult(
        entities=[_ExtractedEntity(name="Mr. Bingley", entity_type="character", chunk_indices=[1])],
        relations=[],
    )
    nothing_more = _ExtractionResult(found_more=False, entities=[], relations=[])
    client = _StubClient([first, nothing_more])
    extractor = ClaudeEntityExtractor(client=client, gleaning_rounds=1)

    extractor.extract(chunks)

    second_call_messages = client.messages.calls[1]["messages"]
    assert len(second_call_messages) == 3  # original user turn, assistant recap, gleaning prompt
    assert "Mr. Bingley" in second_call_messages[1]["content"]
    assert second_call_messages[1]["role"] == "assistant"


def test_gleaned_relation_can_reference_an_entity_from_an_earlier_round() -> None:
    chunks = [_chunk(0, "hash:0", "Mr. Bingley danced with Jane.")]
    first = _ExtractionResult(
        entities=[_ExtractedEntity(name="Mr. Bingley", entity_type="character", chunk_indices=[1])],
        relations=[],
    )
    gleaned = _ExtractionResult(
        found_more=True,
        entities=[_ExtractedEntity(name="Jane", entity_type="character", chunk_indices=[1])],
        relations=[
            _ExtractedRelation(
                source_name="Mr. Bingley", target_name="Jane",
                relation_type="danced_with", chunk_indices=[1],
            )
        ],
    )
    client = _StubClient([first, gleaned])
    extractor = ClaudeEntityExtractor(client=client, gleaning_rounds=1)

    _entities, relations = extractor.extract(chunks)

    assert len(relations) == 1
    assert relations[0].relation_type == "danced_with"
