"""ClaudeRuleCandidateDetector / ClaudeRuleExtractor / ClaudeRuleVerifier:
hallucinated-index handling, ontology-normalization-failure dropping a
rule rather than storing a weakened one, the single-fact vs. multi-fact
Condition shape, and the verifier's domain-mismatch guard and
no-verdict-means-unsupported rule. No API calls: a stub client stands
in for anthropic.Anthropic, matching tests/test_claude_claim_extractor.py's
shape.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

pytest.importorskip("anthropic")

import anthropic
import httpx

from ragcore.errors import GenerationError
from ragcore.rules.base import CandidateSpan
from ragcore.rules.claude import (
    ClaudeRuleCandidateDetector,
    ClaudeRuleExtractor,
    ClaudeRuleVerifier,
    _CandidateDetectionResult,
    _CandidateSpanOut,
    _ExtractedEffect,
    _ExtractedFact,
    _ExtractedRule,
    _RuleExtractionResult,
    _RuleVerificationOut,
    _RuleVerificationResult,
)
from ragcore.rules.types import ConditionGroup, Effect, FactPredicate, Rule
from ragcore.types import Chunk, ChunkMeta


class _StubUsage:
    input_tokens = 10
    output_tokens = 5


class _StubResponse:
    def __init__(self, parsed_output: Any) -> None:
        self.parsed_output = parsed_output
        self.usage = _StubUsage()


class _StubMessages:
    def __init__(self, result: Any) -> None:
        self._result = result
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse(self._result)


class _StubClient:
    def __init__(self, result: Any) -> None:
        self.messages = _StubMessages(result)


class _FakeOntology:
    _KEYS: ClassVar[dict[str, str]] = {"mars house": "planet.mars.house", "mars retrograde": "planet.mars.retrograde"}
    _VALUES: ClassVar[dict[tuple[str, str], int]] = {("planet.mars.house", "8th"): 8}

    def normalize_fact_key(self, raw_mention: str) -> str | None:
        return self._KEYS.get(raw_mention)

    def normalize_fact_value(self, fact_key: str, raw_mention: str) -> Any | None:
        return self._VALUES.get((fact_key, raw_mention))


class _FakeDomainPack:
    domain = "test_domain"
    pack_id = "test_pack"
    pack_version = "1"
    ontology = _FakeOntology()

    def fact_types(self):
        return ()

    def extraction_system_prompt(self) -> str:
        return "extract rules"

    def verification_system_prompt(self) -> str:
        return "verify rules"


def _chunk(chunk_id: str, source_id: str, text: str) -> Chunk:
    meta = ChunkMeta(
        source_id=source_id, source_path="book.md", source_title="Book", source_type="md",
        chapter_index=0, chapter_title="Ch 1", section=None, section_path=(), page_start=None,
        page_end=None, chunk_index=0, context_group_id=f"{source_id}:0", char_start=0,
        char_end=len(text), token_count=10,
    )
    return Chunk(chunk_id=chunk_id, text=text, vector=None, meta=meta)


# -- ClaudeRuleCandidateDetector ------------------------------------------


def test_detector_resolves_a_candidate_to_its_chunk() -> None:
    result = _CandidateDetectionResult(
        candidates=[_CandidateSpanOut(chunk_index=1, text="Mars in the 8th house causes trouble.")]
    )
    client = _StubClient(result)
    detector = ClaudeRuleCandidateDetector(client=client)
    chunks = [_chunk("src:0", "src", "Mars in the 8th house causes trouble.")]

    candidates = detector.detect(chunks, domain_pack=_FakeDomainPack())

    assert len(candidates) == 1
    assert candidates[0].chunk_id == "src:0"
    assert candidates[0].source_id == "src"
    assert candidates[0].text == "Mars in the 8th house causes trouble."


def test_detector_drops_a_hallucinated_chunk_index() -> None:
    result = _CandidateDetectionResult(candidates=[_CandidateSpanOut(chunk_index=99, text="x")])
    client = _StubClient(result)
    detector = ClaudeRuleCandidateDetector(client=client)

    candidates = detector.detect([_chunk("src:0", "src", "text")], domain_pack=_FakeDomainPack())

    assert candidates == []


def test_detector_returns_nothing_for_empty_chunks() -> None:
    detector = ClaudeRuleCandidateDetector(client=_StubClient(_CandidateDetectionResult(candidates=[])))
    assert detector.detect([], domain_pack=_FakeDomainPack()) == []


def test_detector_records_token_usage_from_the_real_response() -> None:
    result = _CandidateDetectionResult(candidates=[])
    detector = ClaudeRuleCandidateDetector(client=_StubClient(result))
    assert detector.usage.total_requests == 0
    assert detector.usage.total_tokens == 0

    detector.detect([_chunk("src:0", "src", "text")], domain_pack=_FakeDomainPack())

    assert detector.usage.total_requests == 1
    assert detector.usage.total_tokens == 15  # _StubUsage: 10 input + 5 output

    detector.detect([_chunk("src:0", "src", "text")], domain_pack=_FakeDomainPack())

    assert detector.usage.total_requests == 2
    assert detector.usage.total_tokens == 30


# -- ClaudeRuleExtractor ---------------------------------------------------


def _candidate(text: str = "Mars in the 8th house causes trouble.") -> CandidateSpan:
    return CandidateSpan(chunk_id="src:0", source_id="src", text=text)


def test_extractor_builds_a_single_predicate_condition_for_one_fact() -> None:
    result = _RuleExtractionResult(
        rules=[
            _ExtractedRule(
                chunk_index=1,
                verbatim_span="Mars in the 8th house causes trouble.",
                facts=[_ExtractedFact(fact_mention="mars house", operator="eq", value_mention="8th")],
                effect=_ExtractedEffect(category="health", polarity="negative"),
                rule_type="HARD",
                confidence=0.9,
            )
        ]
    )
    extractor = ClaudeRuleExtractor(client=_StubClient(result))

    rules = extractor.extract([_candidate()], domain_pack=_FakeDomainPack())

    assert len(rules) == 1
    rule = rules[0]
    assert isinstance(rule.conditions, FactPredicate)
    assert rule.conditions.fact == "planet.mars.house"
    assert rule.conditions.value == 8
    assert rule.domain == "test_domain"
    assert rule.book_id == "src"
    assert rule.chunk_ids == ("src:0",)


def test_extractor_wraps_multiple_facts_in_an_and_group() -> None:
    result = _RuleExtractionResult(
        rules=[
            _ExtractedRule(
                chunk_index=1,
                verbatim_span="Mars in the 8th house, retrograde, causes trouble.",
                facts=[
                    _ExtractedFact(fact_mention="mars house", operator="eq", value_mention="8th"),
                    _ExtractedFact(fact_mention="mars retrograde", operator="exists"),
                ],
                effect=_ExtractedEffect(category="health", polarity="negative"),
                rule_type="HARD",
                confidence=0.9,
            )
        ]
    )
    extractor = ClaudeRuleExtractor(client=_StubClient(result))

    rules = extractor.extract([_candidate()], domain_pack=_FakeDomainPack())

    assert len(rules) == 1
    assert isinstance(rules[0].conditions, ConditionGroup)
    assert rules[0].conditions.op == "and"
    assert len(rules[0].conditions.children) == 2


def test_extractor_drops_a_rule_with_an_unresolvable_fact_key() -> None:
    result = _RuleExtractionResult(
        rules=[
            _ExtractedRule(
                chunk_index=1,
                verbatim_span="Something unrecognized happens.",
                facts=[_ExtractedFact(fact_mention="totally unknown fact", operator="exists")],
                effect=_ExtractedEffect(category="health", polarity="negative"),
                rule_type="HARD",
                confidence=0.9,
            )
        ]
    )
    extractor = ClaudeRuleExtractor(client=_StubClient(result))

    rules = extractor.extract([_candidate()], domain_pack=_FakeDomainPack())

    assert rules == []


def test_extractor_drops_a_rule_with_an_unresolvable_fact_value() -> None:
    result = _RuleExtractionResult(
        rules=[
            _ExtractedRule(
                chunk_index=1,
                verbatim_span="Mars in some unrecognized house causes trouble.",
                facts=[_ExtractedFact(fact_mention="mars house", operator="eq", value_mention="unrecognized")],
                effect=_ExtractedEffect(category="health", polarity="negative"),
                rule_type="HARD",
                confidence=0.9,
            )
        ]
    )
    extractor = ClaudeRuleExtractor(client=_StubClient(result))

    rules = extractor.extract([_candidate()], domain_pack=_FakeDomainPack())

    assert rules == []


def test_extractor_drops_a_hallucinated_chunk_index() -> None:
    result = _RuleExtractionResult(
        rules=[
            _ExtractedRule(
                chunk_index=99, verbatim_span="x",
                facts=[_ExtractedFact(fact_mention="mars house", operator="eq", value_mention="8th")],
                effect=_ExtractedEffect(category="health", polarity="negative"),
                rule_type="HARD", confidence=0.9,
            )
        ]
    )
    extractor = ClaudeRuleExtractor(client=_StubClient(result))
    assert extractor.extract([_candidate()], domain_pack=_FakeDomainPack()) == []


def test_extractor_clamps_confidence_into_zero_one() -> None:
    result = _RuleExtractionResult(
        rules=[
            _ExtractedRule(
                chunk_index=1, verbatim_span="Mars in the 8th house causes trouble.",
                facts=[_ExtractedFact(fact_mention="mars house", operator="eq", value_mention="8th")],
                effect=_ExtractedEffect(category="health", polarity="negative"),
                rule_type="HARD", confidence=1.5,
            )
        ]
    )
    extractor = ClaudeRuleExtractor(client=_StubClient(result))
    rules = extractor.extract([_candidate()], domain_pack=_FakeDomainPack())
    assert rules[0].confidence == 1.0


def test_extractor_rule_id_is_deterministic_across_separate_calls() -> None:
    def make_result():
        return _RuleExtractionResult(
            rules=[
                _ExtractedRule(
                    chunk_index=1, verbatim_span="Mars in the 8th house causes trouble.",
                    facts=[_ExtractedFact(fact_mention="mars house", operator="eq", value_mention="8th")],
                    effect=_ExtractedEffect(category="health", polarity="negative"),
                    rule_type="HARD", confidence=0.9,
                )
            ]
        )
    extractor1 = ClaudeRuleExtractor(client=_StubClient(make_result()))
    extractor2 = ClaudeRuleExtractor(client=_StubClient(make_result()))

    rules1 = extractor1.extract([_candidate()], domain_pack=_FakeDomainPack())
    rules2 = extractor2.extract([_candidate()], domain_pack=_FakeDomainPack())

    assert rules1[0].rule_id == rules2[0].rule_id


def test_extractor_records_token_usage() -> None:
    result = _RuleExtractionResult(rules=[])
    extractor = ClaudeRuleExtractor(client=_StubClient(result))

    extractor.extract([_candidate()], domain_pack=_FakeDomainPack())

    assert extractor.usage.total_requests == 1
    assert extractor.usage.total_tokens == 15


# -- ClaudeRuleVerifier -----------------------------------------------------


def _rule(rule_id: str = "rule:abc") -> Rule:
    return Rule(
        rule_id=rule_id, domain="test_domain", pack_id="p", pack_version="1", book_id="src",
        page_start=1, chunk_ids=("src:0",), verbatim_span="Mars in the 8th house causes trouble.",
        language="en", normalized_statement="n",
        conditions=FactPredicate(fact="planet.mars.house", operator="eq", value=8),
        effect=Effect(category="health", polarity="negative"), rule_type="HARD",
        confidence=0.9, extraction_model="m", extraction_version="v1", created_at="now",
    )


def test_verifier_reports_a_supported_rule() -> None:
    rule = _rule()
    result = _RuleVerificationResult(
        verifications=[_RuleVerificationOut(rule_index=1, supported=True, reason="matches")]
    )
    verifier = ClaudeRuleVerifier(client=_StubClient(result))

    verifications = verifier.verify([rule], domain_pack=_FakeDomainPack())

    assert len(verifications) == 1
    assert verifications[0].rule_id == rule.rule_id
    assert verifications[0].supported is True


def test_verifier_treats_an_omitted_verdict_as_unsupported() -> None:
    rule = _rule()
    result = _RuleVerificationResult(verifications=[])  # model addressed nothing
    verifier = ClaudeRuleVerifier(client=_StubClient(result))

    verifications = verifier.verify([rule], domain_pack=_FakeDomainPack())

    assert len(verifications) == 1
    assert verifications[0].supported is False
    assert "did not return a verdict" in verifications[0].reason


def test_verifier_rejects_a_domain_mismatch() -> None:
    rule = _rule()  # domain="test_domain"

    class _OtherDomainPack(_FakeDomainPack):
        domain = "other_domain"

    verifier = ClaudeRuleVerifier(client=_StubClient(_RuleVerificationResult(verifications=[])))

    with pytest.raises(GenerationError, match="domain"):
        verifier.verify([rule], domain_pack=_OtherDomainPack())


def test_verifier_returns_nothing_for_empty_rules() -> None:
    verifier = ClaudeRuleVerifier(client=_StubClient(_RuleVerificationResult(verifications=[])))
    assert verifier.verify([], domain_pack=_FakeDomainPack()) == []


def test_verifier_records_token_usage() -> None:
    rule = _rule()
    result = _RuleVerificationResult(
        verifications=[_RuleVerificationOut(rule_index=1, supported=True, reason="matches")]
    )
    verifier = ClaudeRuleVerifier(client=_StubClient(result))

    verifier.verify([rule], domain_pack=_FakeDomainPack())

    assert verifier.usage.total_requests == 1
    assert verifier.usage.total_tokens == 15


def test_verifier_records_a_failure_on_a_rejected_request() -> None:
    client = _StubClient(_RuleVerificationResult(verifications=[]))

    class _RejectingMessages:
        def parse(self, **kwargs):
            raise anthropic.APIStatusError(
                "bad request", response=httpx.Response(400, request=httpx.Request("POST", "https://x")),
                body=None,
            )

    client.messages = _RejectingMessages()
    verifier = ClaudeRuleVerifier(client=client)

    with pytest.raises(GenerationError):
        verifier.verify([_rule()], domain_pack=_FakeDomainPack())

    assert verifier.usage.failed_requests == 1
    assert verifier.usage.total_requests == 0
