"""ClaudeHierarchySummarizer: sends numbered parts to the model and returns
a HierarchyNode using the caller-supplied label directly (never a model
guess). No API calls: a stub client stands in for anthropic.Anthropic.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from ragcore.hierarchy.claude import ClaudeHierarchySummarizer, _HierarchySummaryResult


class _StubResponse:
    def __init__(self, parsed_output: _HierarchySummaryResult) -> None:
        self.parsed_output = parsed_output


class _StubMessages:
    def __init__(self, result: _HierarchySummaryResult) -> None:
        self._result = result
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse(self._result)


class _StubClient:
    def __init__(self, result: _HierarchySummaryResult) -> None:
        self.messages = _StubMessages(result)


def test_uses_the_given_label_directly_not_a_model_guess() -> None:
    client = _StubClient(_HierarchySummaryResult(summary="a synthesized summary"))
    summarizer = ClaudeHierarchySummarizer(client=client)

    node = summarizer.summarize(
        node_id="hierarchy:chapter:src:0", level="chapter", source_id="src",
        label="Chapter One: The Beginning", child_texts=["part one", "part two"],
        child_ids=["c1", "c2"], source_chunk_ids=["c1", "c2"],
    )

    assert node.label == "Chapter One: The Beginning"
    assert node.summary == "a synthesized summary"
    assert node.child_ids == ("c1", "c2")
    assert node.source_chunk_ids == ("c1", "c2")
    assert node.vector is None  # embedding happens in the pipeline, not here


def test_sends_numbered_parts_in_the_prompt() -> None:
    client = _StubClient(_HierarchySummaryResult(summary="s"))
    summarizer = ClaudeHierarchySummarizer(client=client)

    summarizer.summarize(
        node_id="n", level="section", source_id="src", label="Section A",
        child_texts=["alpha text", "beta text"], child_ids=["c1", "c2"],
        source_chunk_ids=["c1", "c2"],
    )

    sent_content = client.messages.calls[0]["messages"][0]["content"]
    assert "[1]" in sent_content and "alpha text" in sent_content
    assert "[2]" in sent_content and "beta text" in sent_content
