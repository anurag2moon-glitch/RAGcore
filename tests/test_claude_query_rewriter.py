"""HistoryAwareQueryRewriter: the no-API-call fast path for a trivial
transcript, prompt construction for a real one, and structured-output
parsing. No API calls: a stub client stands in for anthropic.Anthropic.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from ragcore.conversation.rewriting import HistoryAwareQueryRewriter, _RewrittenQuery
from ragcore.conversation.types import Turn


class _StubResponse:
    def __init__(self, parsed_output: _RewrittenQuery) -> None:
        self.parsed_output = parsed_output


class _StubMessages:
    def __init__(self, result: _RewrittenQuery) -> None:
        self._result = result
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse(self._result)


class _StubClient:
    def __init__(self, result: _RewrittenQuery) -> None:
        self.messages = _StubMessages(result)


def _turn(speaker: str, text: str) -> Turn:
    return Turn(speaker=speaker, text=text, created_at="now")


def test_empty_transcript_returns_empty_string_without_calling_the_api() -> None:
    client = _StubClient(_RewrittenQuery(query="unused"))
    rewriter = HistoryAwareQueryRewriter(client=client)

    assert rewriter.rewrite(transcript=[], speaking_as="socrates") == ""
    assert client.messages.calls == []


def test_single_turn_transcript_returns_that_turns_text_unchanged_without_calling_the_api() -> None:
    client = _StubClient(_RewrittenQuery(query="unused"))
    rewriter = HistoryAwareQueryRewriter(client=client)

    result = rewriter.rewrite(transcript=[_turn("user", "what is virtue?")], speaking_as="socrates")

    assert result == "what is virtue?"
    assert client.messages.calls == []


def test_multi_turn_transcript_calls_the_api_and_returns_the_rewritten_query() -> None:
    client = _StubClient(_RewrittenQuery(query="What does virtue require when one feels guilt?"))
    rewriter = HistoryAwareQueryRewriter(client=client)
    transcript = [
        _turn("user", "what is virtue?"),
        _turn("socrates", "virtue is knowledge of the good."),
        _turn("user", "what about when I feel guilty about it?"),
    ]

    result = rewriter.rewrite(transcript=transcript, speaking_as="socrates")

    assert result == "What does virtue require when one feels guilt?"
    assert len(client.messages.calls) == 1


def test_prompt_names_the_speaking_persona_and_includes_the_full_transcript() -> None:
    client = _StubClient(_RewrittenQuery(query="q"))
    rewriter = HistoryAwareQueryRewriter(client=client)
    transcript = [_turn("user", "hello there"), _turn("socrates", "greetings, friend")]

    rewriter.rewrite(transcript=transcript, speaking_as="socrates")

    sent_content = client.messages.calls[0]["messages"][0]["content"]
    assert "socrates is about to respond next" in sent_content
    assert "user: hello there" in sent_content
    assert "socrates: greetings, friend" in sent_content
