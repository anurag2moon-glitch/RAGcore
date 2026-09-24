"""ClaudeClaimLinker: numbered-claim prompt construction, relation-id
canonicalization on the way out, self-pair and hallucinated-index dropping,
and duplicate-relation collapsing within one call. No API calls: a stub
client stands in for anthropic.Anthropic.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from ragcore.claims.claude import ClaudeClaimLinker, _ClaimLinkResult, _ExtractedRelation
from ragcore.claims.types import SourceClaim


class _StubResponse:
    def __init__(self, parsed_output: _ClaimLinkResult) -> None:
        self.parsed_output = parsed_output


class _StubMessages:
    def __init__(self, result: _ClaimLinkResult) -> None:
        self._result = result
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse(self._result)


class _StubClient:
    def __init__(self, result: _ClaimLinkResult) -> None:
        self.messages = _StubMessages(result)


def _claim(claim_id: str, source_id: str, text: str, confidence: str = "firm") -> SourceClaim:
    return SourceClaim(
        claim_id=claim_id, concept_ids=("concept:x",), source_id=source_id, chunk_id=f"{source_id}:0",
        text=text, author_confidence=confidence, model="fake", prompt_version="v1", created_at="now",
    )


def test_links_two_claims_and_canonicalizes_the_relation_id() -> None:
    result = _ClaimLinkResult(
        relations=[_ExtractedRelation(source_index=1, target_index=2, relation_type="conflicts",
                                       rationale="opposing views")]
    )
    client = _StubClient(result)
    linker = ClaudeClaimLinker(client=client)
    claims = [_claim("claim:a", "srcA", "text a"), _claim("claim:b", "srcB", "text b", "hedged")]

    relations = linker.link(claims)

    assert len(relations) == 1
    rel = relations[0]
    assert rel.relation_type == "conflicts"
    assert {rel.source_claim_id, rel.target_claim_id} == {"claim:a", "claim:b"}
    assert rel.rationale == "opposing views"


def test_extends_relation_preserves_reported_direction() -> None:
    result = _ClaimLinkResult(
        relations=[_ExtractedRelation(source_index=2, target_index=1, relation_type="extends", rationale="r")]
    )
    client = _StubClient(result)
    linker = ClaudeClaimLinker(client=client)
    claims = [_claim("claim:a", "srcA", "text a"), _claim("claim:b", "srcB", "text b")]

    relations = linker.link(claims)

    assert relations[0].source_claim_id == "claim:b"
    assert relations[0].target_claim_id == "claim:a"


def test_self_pairs_are_dropped() -> None:
    result = _ClaimLinkResult(
        relations=[_ExtractedRelation(source_index=1, target_index=1, relation_type="agrees", rationale="r")]
    )
    client = _StubClient(result)
    linker = ClaudeClaimLinker(client=client)
    claims = [_claim("claim:a", "srcA", "text a"), _claim("claim:b", "srcB", "text b")]

    assert linker.link(claims) == []


def test_hallucinated_index_is_dropped() -> None:
    result = _ClaimLinkResult(
        relations=[_ExtractedRelation(source_index=1, target_index=99, relation_type="agrees", rationale="r")]
    )
    client = _StubClient(result)
    linker = ClaudeClaimLinker(client=client)
    claims = [_claim("claim:a", "srcA", "text a"), _claim("claim:b", "srcB", "text b")]

    assert linker.link(claims) == []


def test_duplicate_relations_within_one_call_collapse_to_one() -> None:
    result = _ClaimLinkResult(
        relations=[
            _ExtractedRelation(source_index=1, target_index=2, relation_type="agrees", rationale="r1"),
            _ExtractedRelation(source_index=2, target_index=1, relation_type="agrees", rationale="r2"),
        ]
    )
    client = _StubClient(result)
    linker = ClaudeClaimLinker(client=client)
    claims = [_claim("claim:a", "srcA", "text a"), _claim("claim:b", "srcB", "text b")]

    relations = linker.link(claims)

    assert len(relations) == 1  # same unordered pair, same type -> same relation_id


def test_fewer_than_two_claims_returns_nothing_without_calling_the_api() -> None:
    client = _StubClient(_ClaimLinkResult(relations=[]))
    linker = ClaudeClaimLinker(client=client)

    assert linker.link([_claim("claim:a", "srcA", "text a")]) == []
    assert client.messages.calls == []


def test_sends_numbered_claims_with_source_and_confidence_in_the_prompt() -> None:
    client = _StubClient(_ClaimLinkResult(relations=[]))
    linker = ClaudeClaimLinker(client=client)
    claims = [_claim("claim:a", "srcA", "alpha claim text", "firm"), _claim("claim:b", "srcB", "beta claim text", "hedged")]

    linker.link(claims)

    sent_content = client.messages.calls[0]["messages"][0]["content"]
    assert "[1]" in sent_content and "alpha claim text" in sent_content and "firm" in sent_content
    assert "[2]" in sent_content and "beta claim text" in sent_content and "hedged" in sent_content
