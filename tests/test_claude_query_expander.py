"""ClaudeQueryExpander: one call producing HyDE passage + alternative
phrasings + sub-questions, deduped and capped — no API calls, a stub
client stands in for anthropic.Anthropic (same pattern as
test_claude_sufficiency_judge.py).
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from ragcore.retrieval.query_expansion import ClaudeQueryExpander, _ExpansionResult


class _StubResponse:
    def __init__(self, parsed_output: _ExpansionResult) -> None:
        self.parsed_output = parsed_output


class _StubMessages:
    def __init__(self, result: _ExpansionResult) -> None:
        self._result = result
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse(self._result)


class _StubClient:
    def __init__(self, result: _ExpansionResult) -> None:
        self.messages = _StubMessages(result)


def test_expand_combines_hyde_phrasings_and_sub_questions_in_order() -> None:
    result = _ExpansionResult(
        hyde_passage="A hypothetical passage about the topic.",
        alternative_phrasings=["a different angle"],
        sub_questions=["first sub-question?", "second sub-question?"],
    )
    client = _StubClient(result)
    expander = ClaudeQueryExpander(client=client)

    variants = expander.expand("original query")

    assert variants == [
        "A hypothetical passage about the topic.",
        "a different angle",
        "first sub-question?",
        "second sub-question?",
    ]


def test_expand_dedupes_while_preserving_order() -> None:
    result = _ExpansionResult(
        hyde_passage="same text",
        alternative_phrasings=["same text", "a unique one"],
        sub_questions=[],
    )
    client = _StubClient(result)
    expander = ClaudeQueryExpander(client=client)

    variants = expander.expand("q")

    assert variants == ["same text", "a unique one"]


def test_expand_strips_and_drops_empty_variants() -> None:
    result = _ExpansionResult(
        hyde_passage="  real passage  ",
        alternative_phrasings=["", "   "],
        sub_questions=[],
    )
    client = _StubClient(result)
    expander = ClaudeQueryExpander(client=client)

    variants = expander.expand("q")

    assert variants == ["real passage"]


def test_expand_caps_at_max_variants() -> None:
    result = _ExpansionResult(
        hyde_passage="hyde",
        alternative_phrasings=["a1", "a2", "a3"],
        sub_questions=["s1", "s2", "s3"],
    )
    client = _StubClient(result)
    expander = ClaudeQueryExpander(client=client, max_variants=3)

    variants = expander.expand("q")

    assert variants == ["hyde", "a1", "a2"]


def test_sends_the_query_in_the_request() -> None:
    result = _ExpansionResult(hyde_passage="hyde", alternative_phrasings=[], sub_questions=[])
    client = _StubClient(result)
    expander = ClaudeQueryExpander(client=client)

    expander.expand("what caused the fall of the empire?")

    sent_content = client.messages.calls[0]["messages"][0]["content"]
    assert "what caused the fall of the empire?" in sent_content
