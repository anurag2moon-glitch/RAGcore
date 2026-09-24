"""ClaudeVerifier and VerifyingGenerator: supported/unsupported/
supported_by_synthesis verdicts against a stub Anthropic client (same
_StubResponse/_StubMessages/_StubClient shape as
test_claude_extractor_citations.py), and the flag-don't-strip guarantee —
Answer.text must come through byte-identical, never rewritten.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from ragcore.generation.base import Answer
from ragcore.types import Chunk, ChunkMeta, Retrieved
from ragcore.verification.claude import (
    ClaudeVerifier,
    VerifyingGenerator,
    _CheckedClaim,
    _CheckResult,
    _ClaimList,
)


def _chunk(chunk_id: str, text: str, *, synthetic: bool = False) -> Chunk:
    meta = ChunkMeta(
        source_id="src", source_path="p", source_title="t",
        source_type="synthetic" if synthetic else "md",
        section_path=(), chapter_title=None, chapter_index=None, section=None,
        page_start=None, page_end=None, chunk_index=0, context_group_id="g",
        char_start=0, char_end=len(text), token_count=len(text) // 4,
        extra={"ragcore_kind": "community_summary"} if synthetic else {},
    )
    return Chunk(chunk_id=chunk_id, text=text, meta=meta)


class _StubResponse:
    def __init__(self, parsed_output) -> None:
        self.parsed_output = parsed_output


class _StubMessages:
    def __init__(self, results: list) -> None:
        self._results = results
        self.calls: list[dict] = []
        self._i = 0

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        result = self._results[self._i]
        self._i += 1
        return _StubResponse(result)


class _StubClient:
    def __init__(self, results: list) -> None:
        self.messages = _StubMessages(results)


def test_supported_and_unsupported_claims_are_distinguished() -> None:
    real = Retrieved(chunk=_chunk("h:0", "The sky is blue."), score=0.9)
    answer = Answer(text="The sky is blue. Aliens built the pyramids.", citations=[real])

    decompose = _ClaimList(claims=["The sky is blue.", "Aliens built the pyramids."])
    check = _CheckResult(
        checks=[
            _CheckedClaim(claim_index=1, supporting_indices=[1]),
            _CheckedClaim(claim_index=2, supporting_indices=[]),
        ]
    )
    client = _StubClient([decompose, check])
    verifier = ClaudeVerifier(client=client)

    verification = verifier.verify("what color is the sky", answer)

    assert verification.checks[0].status == "supported"
    assert verification.checks[0].support == ("h:0",)
    assert verification.checks[1].status == "unsupported"
    assert verification.checks[1].support == ()
    assert verification.supported_ratio == 0.5
    assert [c.claim for c in verification.unsupported] == ["Aliens built the pyramids."]


def test_status_is_supported_by_synthesis_when_only_a_summary_supports_it() -> None:
    """The verdict is derived mechanically from provenance (ragcore_kind),
    never from the model's own claimed classification — the check call only
    ever returns which numbered items support a claim."""
    summary = Retrieved(chunk=_chunk("community:x", "Summary text.", synthetic=True), score=0.8)
    answer = Answer(text="A synthesized theme claim.", citations=[summary])

    decompose = _ClaimList(claims=["A synthesized theme claim."])
    check = _CheckResult(checks=[_CheckedClaim(claim_index=1, supporting_indices=[1])])
    client = _StubClient([decompose, check])
    verifier = ClaudeVerifier(client=client)

    verification = verifier.verify("q", answer)

    assert verification.checks[0].status == "supported_by_synthesis"


def test_status_is_supported_not_synthesis_when_a_real_passage_also_supports_it() -> None:
    real = Retrieved(chunk=_chunk("h:0", "Real passage."), score=0.9)
    summary = Retrieved(chunk=_chunk("community:x", "Summary text.", synthetic=True), score=0.8)
    answer = Answer(text="A claim backed by both.", citations=[real, summary])

    decompose = _ClaimList(claims=["A claim backed by both."])
    check = _CheckResult(checks=[_CheckedClaim(claim_index=1, supporting_indices=[1, 2])])
    client = _StubClient([decompose, check])
    verifier = ClaudeVerifier(client=client)

    verification = verifier.verify("q", answer)

    assert verification.checks[0].status == "supported"


def test_out_of_range_supporting_index_is_dropped_not_crash() -> None:
    real = Retrieved(chunk=_chunk("h:0", "Real passage."), score=0.9)
    answer = Answer(text="A claim.", citations=[real])

    decompose = _ClaimList(claims=["A claim."])
    check = _CheckResult(checks=[_CheckedClaim(claim_index=1, supporting_indices=[1, 99])])
    client = _StubClient([decompose, check])
    verifier = ClaudeVerifier(client=client)

    verification = verifier.verify("q", answer)

    assert verification.checks[0].support == ("h:0",)


def test_no_claims_decomposed_yields_no_checks_without_a_second_call() -> None:
    real = Retrieved(chunk=_chunk("h:0", "Nothing checkable here."), score=0.9)
    answer = Answer(text="Nothing checkable here.", citations=[real])

    client = _StubClient([_ClaimList(claims=[])])
    verifier = ClaudeVerifier(client=client)

    verification = verifier.verify("q", answer)

    assert verification.checks == ()
    assert len(client.messages.calls) == 1  # only the decompose call, never a check call


def test_verifying_generator_flags_but_never_strips_the_answer_text() -> None:
    class _FakeInnerGenerator:
        model = "fake"

        def generate(self, query, context):
            return Answer(text="The sky is blue. Aliens built the pyramids.", citations=list(context))

    real = Retrieved(chunk=_chunk("h:0", "The sky is blue."), score=0.9)
    decompose = _ClaimList(claims=["The sky is blue.", "Aliens built the pyramids."])
    check = _CheckResult(
        checks=[
            _CheckedClaim(claim_index=1, supporting_indices=[1]),
            _CheckedClaim(claim_index=2, supporting_indices=[]),
        ]
    )
    client = _StubClient([decompose, check])
    verifier = ClaudeVerifier(client=client)
    generator = VerifyingGenerator(_FakeInnerGenerator(), verifier)

    answer = generator.generate("q", [real])

    assert answer.text == "The sky is blue. Aliens built the pyramids."  # unchanged
    assert answer.verification is not None
    assert len(answer.verification.unsupported) == 1


def test_plain_generator_never_sets_verification() -> None:
    from tests.fakes import FakeGenerator

    gen = FakeGenerator()
    answer = gen.generate("q", [])
    assert answer.verification is None
