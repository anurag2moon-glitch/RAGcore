"""ClaudeConceptExtractor's opt-in gleaning_rounds (docs/ARCHITECTURE.md
Phase 4 gap list item 17, concepts half) — no API calls, a stub client
returns a scripted sequence of responses, one per call.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from ragcore.concepts.claude_extractor import (
    ClaudeConceptExtractor,
    _ConceptResult,
    _ExtractedConcept,
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
    def __init__(self, parsed_output: _ConceptResult) -> None:
        self.parsed_output = parsed_output


class _ScriptedMessages:
    def __init__(self, results: list[_ConceptResult]) -> None:
        self._results = results
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse(self._results[len(self.calls) - 1])


class _StubClient:
    def __init__(self, results: list[_ConceptResult]) -> None:
        self.messages = _ScriptedMessages(results)


def test_zero_gleaning_rounds_makes_exactly_one_call() -> None:
    chunks = [_chunk(0, "hash:0", "text about ambition")]
    result = _ConceptResult(
        concepts=[_ExtractedConcept(label="consequences of unchecked ambition", description="d", chunk_indices=[1])]
    )
    client = _StubClient([result])
    extractor = ClaudeConceptExtractor(client=client)

    concepts, _edges = extractor.extract(chunks)

    assert len(client.messages.calls) == 1
    assert [c.label for c in concepts] == ["consequences of unchecked ambition"]


def test_gleaning_round_adds_a_newly_found_concept() -> None:
    chunks = [_chunk(0, "hash:0", "text")]
    first = _ConceptResult(
        concepts=[_ExtractedConcept(label="consequences of unchecked ambition", description="d1", chunk_indices=[1])]
    )
    gleaned = _ConceptResult(
        found_more=True,
        concepts=[_ExtractedConcept(label="tension between duty and desire", description="d2", chunk_indices=[1])],
    )
    client = _StubClient([first, gleaned])
    extractor = ClaudeConceptExtractor(client=client, gleaning_rounds=1)

    concepts, _edges = extractor.extract(chunks)

    assert len(client.messages.calls) == 2
    assert {c.label for c in concepts} == {
        "consequences of unchecked ambition",
        "tension between duty and desire",
    }


def test_gleaning_stops_early_when_model_reports_nothing_more() -> None:
    chunks = [_chunk(0, "hash:0", "text")]
    first = _ConceptResult(
        concepts=[_ExtractedConcept(label="consequences of unchecked ambition", description="d", chunk_indices=[1])]
    )
    nothing_more = _ConceptResult(found_more=False, concepts=[])
    client = _StubClient([first, nothing_more, nothing_more])
    extractor = ClaudeConceptExtractor(client=client, gleaning_rounds=3)

    extractor.extract(chunks)

    assert len(client.messages.calls) == 2


def test_gleaning_round_merges_chunk_ids_for_a_concept_re_found_under_the_same_label() -> None:
    chunks = [_chunk(0, "hash:0", "a"), _chunk(1, "hash:1", "b")]
    first = _ConceptResult(
        concepts=[_ExtractedConcept(label="consequences of unchecked ambition", description="d", chunk_indices=[1])]
    )
    gleaned = _ConceptResult(
        found_more=True,
        concepts=[_ExtractedConcept(label="consequences of unchecked ambition", description="d", chunk_indices=[2])],
    )
    client = _StubClient([first, gleaned])
    extractor = ClaudeConceptExtractor(client=client, gleaning_rounds=1)

    [concept] = extractor.extract(chunks)[0]

    assert set(concept.source_chunk_ids) == {"hash:0", "hash:1"}


def test_gleaning_conversation_includes_a_recap_of_prior_findings() -> None:
    chunks = [_chunk(0, "hash:0", "text")]
    first = _ConceptResult(
        concepts=[_ExtractedConcept(label="consequences of unchecked ambition", description="d", chunk_indices=[1])]
    )
    nothing_more = _ConceptResult(found_more=False, concepts=[])
    client = _StubClient([first, nothing_more])
    extractor = ClaudeConceptExtractor(client=client, gleaning_rounds=1)

    extractor.extract(chunks)

    second_call_messages = client.messages.calls[1]["messages"]
    assert len(second_call_messages) == 3
    assert "consequences of unchecked ambition" in second_call_messages[1]["content"]
    assert second_call_messages[1]["role"] == "assistant"
