"""ClaudeClaimExtractor: numbered chunk/concept prompt construction,
hallucinated-index handling (chunk and concept), the
zero-resolved-concepts-drops-the-claim rule, and author_confidence
round-tripping. No API calls: a stub client stands in for anthropic.Anthropic.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from ragcore.claims.claude import ClaudeClaimExtractor, _ClaimExtractionResult, _ExtractedClaim
from ragcore.concepts.types import Concept
from ragcore.types import Chunk, ChunkMeta


class _StubResponse:
    def __init__(self, parsed_output: _ClaimExtractionResult) -> None:
        self.parsed_output = parsed_output


class _StubMessages:
    def __init__(self, result: _ClaimExtractionResult) -> None:
        self._result = result
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse(self._result)


class _StubClient:
    def __init__(self, result: _ClaimExtractionResult) -> None:
        self.messages = _StubMessages(result)


def _chunk(chunk_id: str, source_id: str, text: str) -> Chunk:
    meta = ChunkMeta(
        source_id=source_id, source_path="book.md", source_title="Book", source_type="md",
        chapter_index=0, chapter_title="Ch 1", section=None, section_path=(), page_start=None,
        page_end=None, chunk_index=0, context_group_id=f"{source_id}:0", char_start=0,
        char_end=len(text), token_count=10,
    )
    return Chunk(chunk_id=chunk_id, text=text, vector=None, meta=meta)


def _concept(concept_id: str, label: str) -> Concept:
    return Concept(concept_id=concept_id, label=label, description="d", source_chunk_ids=())


def test_extracts_a_claim_with_resolved_chunk_and_concept() -> None:
    result = _ClaimExtractionResult(
        claims=[_ExtractedClaim(chunk_index=1, concept_indices=[1], text="Freedom requires struggle.",
                                 author_confidence="firm")]
    )
    client = _StubClient(result)
    extractor = ClaudeClaimExtractor(client=client)
    chunks = [_chunk("src:0", "src", "Freedom requires struggle against weakness.")]
    concepts = [_concept("concept:freedom", "freedom")]

    claims = extractor.extract(chunks=chunks, concepts=concepts)

    assert len(claims) == 1
    assert claims[0].text == "Freedom requires struggle."
    assert claims[0].concept_ids == ("concept:freedom",)
    assert claims[0].chunk_id == "src:0"
    assert claims[0].source_id == "src"
    assert claims[0].author_confidence == "firm"


def test_hallucinated_chunk_index_drops_the_claim() -> None:
    result = _ClaimExtractionResult(
        claims=[_ExtractedClaim(chunk_index=99, concept_indices=[1], text="x", author_confidence="firm")]
    )
    client = _StubClient(result)
    extractor = ClaudeClaimExtractor(client=client)
    claims = extractor.extract(
        chunks=[_chunk("src:0", "src", "text")], concepts=[_concept("concept:x", "x")]
    )
    assert claims == []


def test_a_claim_with_only_hallucinated_concept_indices_is_dropped() -> None:
    result = _ClaimExtractionResult(
        claims=[_ExtractedClaim(chunk_index=1, concept_indices=[99], text="x", author_confidence="firm")]
    )
    client = _StubClient(result)
    extractor = ClaudeClaimExtractor(client=client)
    claims = extractor.extract(
        chunks=[_chunk("src:0", "src", "text")], concepts=[_concept("concept:x", "x")]
    )
    assert claims == []


def test_a_claim_with_partially_hallucinated_concept_indices_keeps_the_valid_ones() -> None:
    result = _ClaimExtractionResult(
        claims=[_ExtractedClaim(chunk_index=1, concept_indices=[1, 99], text="x", author_confidence="hedged")]
    )
    client = _StubClient(result)
    extractor = ClaudeClaimExtractor(client=client)
    claims = extractor.extract(
        chunks=[_chunk("src:0", "src", "text")], concepts=[_concept("concept:x", "x")]
    )
    assert len(claims) == 1
    assert claims[0].concept_ids == ("concept:x",)
    assert claims[0].author_confidence == "hedged"


def test_empty_concepts_or_chunks_returns_no_claims_without_calling_the_api() -> None:
    client = _StubClient(_ClaimExtractionResult(claims=[]))
    extractor = ClaudeClaimExtractor(client=client)

    assert extractor.extract(chunks=[], concepts=[_concept("concept:x", "x")]) == []
    assert extractor.extract(chunks=[_chunk("src:0", "src", "t")], concepts=[]) == []
    assert client.messages.calls == []


def test_sends_numbered_chunks_and_numbered_concepts_in_the_prompt() -> None:
    client = _StubClient(_ClaimExtractionResult(claims=[]))
    extractor = ClaudeClaimExtractor(client=client)

    extractor.extract(
        chunks=[_chunk("src:0", "src", "alpha text")],
        concepts=[_concept("concept:a", "concept alpha")],
    )

    sent_content = client.messages.calls[0]["messages"][0]["content"]
    assert "[1]" in sent_content and "alpha text" in sent_content
    assert "(1)" in sent_content and "concept alpha" in sent_content
