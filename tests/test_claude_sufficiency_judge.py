"""ClaudeSufficiencyJudge: sends numbered context (never real chunk_id
hashes) and correctly parses the model's sufficient/next_query/reasoning
response. No API calls: a stub client stands in for anthropic.Anthropic.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from ragcore.retrieval.iterative import ClaudeSufficiencyJudge, HopRecord, _SufficiencyResult
from ragcore.types import Chunk, ChunkMeta, Retrieved


def _chunk(chunk_id: str, text: str) -> Chunk:
    meta = ChunkMeta(
        source_id="src", source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=None, section=None,
        page_start=None, page_end=None, chunk_index=0, context_group_id="g",
        char_start=0, char_end=len(text), token_count=len(text) // 4,
    )
    return Chunk(chunk_id=chunk_id, text=text, meta=meta)


class _StubResponse:
    def __init__(self, parsed_output: _SufficiencyResult) -> None:
        self.parsed_output = parsed_output


class _StubMessages:
    def __init__(self, result: _SufficiencyResult) -> None:
        self._result = result
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse(self._result)


class _StubClient:
    def __init__(self, result: _SufficiencyResult) -> None:
        self.messages = _StubMessages(result)


def test_sends_numbered_context_never_real_chunk_ids() -> None:
    context = [Retrieved(chunk=_chunk("hash:0", "some passage text"), score=0.9)]
    result = _SufficiencyResult(sufficient=True, next_query=None, reasoning="covers it")
    client = _StubClient(result)
    judge = ClaudeSufficiencyJudge(client=client)

    decision = judge.assess("original question", [], context)

    assert decision.sufficient is True
    assert decision.next_query is None
    sent_content = client.messages.calls[0]["messages"][0]["content"]
    assert "[1]" in sent_content
    assert "hash:0" not in sent_content
    assert "original question" in sent_content


def test_reports_a_next_query_when_insufficient() -> None:
    context = [Retrieved(chunk=_chunk("hash:0", "partial info"), score=0.5)]
    result = _SufficiencyResult(sufficient=False, next_query="a sharper follow-up", reasoning="missing detail")
    client = _StubClient(result)
    judge = ClaudeSufficiencyJudge(client=client)

    decision = judge.assess("original question", [], context)

    assert decision.sufficient is False
    assert decision.next_query == "a sharper follow-up"


def test_previous_hops_are_included_in_the_prompt() -> None:
    context = [Retrieved(chunk=_chunk("hash:0", "text"), score=0.5)]
    hops = [HopRecord(hop_index=0, query="first try", retrieved_chunk_ids=("hash:0",), sufficient=False, reasoning="r")]
    result = _SufficiencyResult(sufficient=True, next_query=None, reasoning="ok")
    client = _StubClient(result)
    judge = ClaudeSufficiencyJudge(client=client)

    judge.assess("original question", hops, context)

    sent_content = client.messages.calls[0]["messages"][0]["content"]
    assert "first try" in sent_content
