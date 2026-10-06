"""Opt-in prompt caching (docs/ARCHITECTURE.md Phase 5 gap list item 21,
prompt-caching half) on ClaudeGenerator, ClaudeEntityExtractor, and
ClaudeConceptExtractor — no API calls, stub clients inspect the exact
request payload sent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from ragcore.prompt_caching import cached_system, cached_text_block


def test_cached_text_block_shape() -> None:
    block = cached_text_block("hello")
    assert block == {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}


def test_cached_system_wraps_in_a_list() -> None:
    assert cached_system("sys") == [cached_text_block("sys")]


# -- ClaudeGenerator --------------------------------------------------------


class _TextBlock:
    type = "text"
    text = "answer"


class _GenStubResponse:
    stop_reason = "end_turn"
    stop_details = None

    def __init__(self) -> None:
        self.content = [_TextBlock()]


class _GenStubMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _GenStubResponse()


class _GenStubClient:
    def __init__(self) -> None:
        self.messages = _GenStubMessages()


def test_generator_system_is_plain_string_by_default() -> None:
    from ragcore.generation.claude import ClaudeGenerator

    client = _GenStubClient()
    ClaudeGenerator(client=client).generate("q", [])

    assert isinstance(client.messages.calls[0]["system"], str)


def test_generator_system_is_cached_when_enabled() -> None:
    from ragcore.generation.claude import ClaudeGenerator

    client = _GenStubClient()
    ClaudeGenerator(client=client, enable_prompt_caching=True).generate("q", [])

    system = client.messages.calls[0]["system"]
    assert system == cached_system(ClaudeGenerator(client=client)._system_prompt)


# -- ClaudeEntityExtractor / ClaudeConceptExtractor ------------------------


def _chunk():
    from ragcore.types import Chunk, ChunkMeta

    meta = ChunkMeta(
        source_id="src", source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=0, section=None,
        page_start=None, page_end=None, chunk_index=0, context_group_id="g",
        char_start=0, char_end=4, token_count=1,
    )
    return Chunk(chunk_id="hash:0", text="text", meta=meta)


class _ParseStubMessages:
    def __init__(self, result) -> None:
        self._result = result
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)

        class _R:
            parsed_output = self._result

        return _R()


class _ParseStubClient:
    def __init__(self, result) -> None:
        self.messages = _ParseStubMessages(result)


def test_entity_extractor_messages_are_plain_strings_by_default() -> None:
    from ragcore.graph.claude_extractor import ClaudeEntityExtractor, _ExtractionResult

    result = _ExtractionResult(entities=[], relations=[])
    client = _ParseStubClient(result)
    ClaudeEntityExtractor(client=client).extract([_chunk()])

    call = client.messages.calls[0]
    assert isinstance(call["system"], str)
    assert isinstance(call["messages"][0]["content"], str)


def test_entity_extractor_caches_system_and_first_turn_when_enabled() -> None:
    from ragcore.graph.claude_extractor import ClaudeEntityExtractor, _ExtractionResult

    result = _ExtractionResult(entities=[], relations=[])
    client = _ParseStubClient(result)
    ClaudeEntityExtractor(client=client, enable_prompt_caching=True).extract([_chunk()])

    call = client.messages.calls[0]
    assert isinstance(call["system"], list)
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    first_turn = call["messages"][0]["content"]
    assert isinstance(first_turn, list)
    assert first_turn[0]["cache_control"] == {"type": "ephemeral"}


def test_concept_extractor_caches_system_and_first_turn_when_enabled() -> None:
    from ragcore.concepts.claude_extractor import ClaudeConceptExtractor, _ConceptResult

    result = _ConceptResult(concepts=[])
    client = _ParseStubClient(result)
    ClaudeConceptExtractor(client=client, enable_prompt_caching=True).extract([_chunk()])

    call = client.messages.calls[0]
    assert isinstance(call["system"], list)
    first_turn = call["messages"][0]["content"]
    assert isinstance(first_turn, list)
    assert first_turn[0]["cache_control"] == {"type": "ephemeral"}


def test_concept_extractor_messages_are_plain_strings_by_default() -> None:
    from ragcore.concepts.claude_extractor import ClaudeConceptExtractor, _ConceptResult

    result = _ConceptResult(concepts=[])
    client = _ParseStubClient(result)
    ClaudeConceptExtractor(client=client).extract([_chunk()])

    call = client.messages.calls[0]
    assert isinstance(call["system"], str)
    assert isinstance(call["messages"][0]["content"], str)
