"""Verifies ClaudeConceptExtractor cites short numbered indices to the model
(never full chunk_id hashes — same rationale as
test_claude_extractor_citations.py), correctly remaps those indices back to
real chunk_ids, and derives co-occurrence edges from which concepts share a
chunk rather than asking the model for edges directly. No API calls: a stub
client stands in for anthropic.Anthropic.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from ragcore.concepts.claude_extractor import (
    ClaudeConceptExtractor,
    _ConceptResult,
    _ExtractedConcept,
    _concept_id,
    _derive_edges,
)
from ragcore.concepts.types import Concept
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
    def __init__(self, parsed_output: _ConceptResult) -> None:
        self.parsed_output = parsed_output


class _StubMessages:
    def __init__(self, result: _ConceptResult) -> None:
        self._result = result
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse(self._result)


class _StubClient:
    def __init__(self, result: _ConceptResult) -> None:
        self.messages = _StubMessages(result)


def test_extract_sends_numeric_labels_and_remaps_response_to_real_chunk_ids() -> None:
    chunks = [
        _chunk(0, "hash:0", "The captain's ambition consumed him."),
        _chunk(1, "hash:1", "His downfall came from that same ambition."),
    ]
    result = _ConceptResult(
        concepts=[
            _ExtractedConcept(
                label="consequences of unchecked ambition",
                description="Ambition pursued without restraint costs the pursuer.",
                chunk_indices=[1, 2],
            ),
        ]
    )
    client = _StubClient(result)
    extractor = ClaudeConceptExtractor(client=client)

    concepts, edges = extractor.extract(chunks)

    assert len(concepts) == 1
    assert concepts[0].concept_id == _concept_id("consequences of unchecked ambition")
    assert concepts[0].source_chunk_ids == ("hash:0", "hash:1")
    assert edges == []  # a single concept has nothing to co-occur with

    sent_content = client.messages.calls[0]["messages"][0]["content"]
    assert "[1]" in sent_content and "[2]" in sent_content
    assert "hash:0" not in sent_content
    assert "hash:1" not in sent_content


def test_extract_drops_out_of_range_indices_without_crashing() -> None:
    chunks = [_chunk(0, "hash:0", "text")]
    result = _ConceptResult(
        concepts=[
            _ExtractedConcept(label="some idea", description="d", chunk_indices=[1, 99]),
        ]
    )
    client = _StubClient(result)
    extractor = ClaudeConceptExtractor(client=client)

    concepts, _edges = extractor.extract(chunks)

    assert concepts[0].source_chunk_ids == ("hash:0",)  # index 99 silently dropped


def test_extract_merges_repeated_label_within_one_chapter_into_one_concept() -> None:
    """The model is instructed to reuse an identical label for a recurring
    idea within one chapter; extract() must collapse those into a single
    Concept with the union of chunk_ids, not one Concept per mention."""
    chunks = [_chunk(0, "hash:0", "a"), _chunk(1, "hash:1", "b")]
    result = _ConceptResult(
        concepts=[
            _ExtractedConcept(label="unchecked ambition", description="d1", chunk_indices=[1]),
            _ExtractedConcept(label="unchecked ambition", description="d2", chunk_indices=[2]),
        ]
    )
    client = _StubClient(result)
    extractor = ClaudeConceptExtractor(client=client)

    concepts, _edges = extractor.extract(chunks)

    assert len(concepts) == 1
    assert concepts[0].source_chunk_ids == ("hash:0", "hash:1")


def test_extract_returns_empty_for_no_chunks() -> None:
    client = _StubClient(_ConceptResult(concepts=[]))
    extractor = ClaudeConceptExtractor(client=client)

    concepts, edges = extractor.extract([])

    assert concepts == []
    assert edges == []
    assert client.messages.calls == []  # no API call made for an empty chapter


def test_derive_edges_counts_distinct_co_occurring_chunks_as_weight() -> None:
    """Two concepts appearing together in two different chunks should
    produce one edge with weight=2, not two separate edges."""
    a = Concept(concept_id="concept:a", label="a", description="d", source_chunk_ids=("c1", "c2"))
    b = Concept(concept_id="concept:b", label="b", description="d", source_chunk_ids=("c1", "c2"))
    c = Concept(concept_id="concept:c", label="c", description="d", source_chunk_ids=("c1",))

    edges = _derive_edges([a, b, c])

    by_id = {e.edge_id: e for e in edges}
    ab_edge = by_id["edge:concept:a:concept:b"]
    assert ab_edge.weight == 2
    assert set(ab_edge.source_chunk_ids) == {"c1", "c2"}

    ac_edge = by_id["edge:concept:a:concept:c"]
    assert ac_edge.weight == 1
    assert ac_edge.source_chunk_ids == ("c1",)
