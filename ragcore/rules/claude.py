"""RuleCandidateDetector, RuleExtractor, and RuleVerifier implementations
using Claude — Phase 10 §2's three passes, mirroring the structured-output
(`client.messages.parse`) + retry shape every other Claude-backed
component in this codebase uses (e.g. `ragcore.claims.claude`).

Pass 2 (`ClaudeRuleExtractor`)'s structured-output schema deliberately
does **not** mirror `Condition`'s own recursive AND/OR/NOT tree: a
reliable structured-output schema needs a fixed shape, and asking the
model to emit an arbitrarily-nested predicate tree is both harder to
get right and harder to verify against a verbatim span than a flat
list of facts. The LLM reports a flat list of `(fact, operator, value)`
mentions for a rule and this module wraps them as a single
`ConditionGroup(op="and", ...)` (or the bare `FactPredicate` itself
when there's only one) — correct for Lal Kitab's own rule style ("Mars
in house 8 AND Saturn retrograde -> ...", conjunctions of simple
facts, never OR/NOT in the source text itself). A domain whose rules
genuinely need OR/NOT at extraction time would need its own extractor;
nothing stops a reviewer from manually constructing a richer
`Condition` tree through the review workflow (`ragcore.rules.review`)
for any individual rule after the fact.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, Sequence

from pydantic import BaseModel

from ragcore.embedding.cost import UsageTracker
from ragcore.errors import GenerationError
from ragcore.graph.claude_extractor import _format_chunks
from ragcore.model_routing import DEFAULT_ROUTING
from ragcore.rules.base import CandidateSpan, VerificationResult
from ragcore.rules.types import (
    ConditionGroup,
    Effect,
    FactPredicate,
    Remedy,
    Rule,
    RuleType,
    rule_content_hash,
)

if TYPE_CHECKING:
    from ragcore.rules.domain import DomainPack
    from ragcore.rules.types import Condition
    from ragcore.types import Chunk

try:
    import anthropic
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    anthropic = None
else:
    _IMPORT_ERROR = None

DETECTOR_PROMPT_VERSION = "rule-detect-v1"
EXTRACTOR_PROMPT_VERSION = "rule-extract-v1"
VERIFIER_PROMPT_VERSION = "rule-verify-v1"


class _CandidateSpanOut(BaseModel):
    chunk_index: int
    text: str


class _CandidateDetectionResult(BaseModel):
    candidates: list[_CandidateSpanOut]


class _ExtractedFact(BaseModel):
    fact_mention: str
    """The raw source-text mention of a fact, in whatever vocabulary
    the source uses (e.g. "mangal ashtam bhava mein" or "Mars in the
    8th house") — normalized against the domain pack's ontology after
    extraction, never guessed at by this model."""
    operator: Literal["eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "exists"]
    value_mention: str | None = None
    """Omitted (None) only when operator == "exists"."""


class _ExtractedEffect(BaseModel):
    category: str
    polarity: Literal["positive", "negative", "neutral"]
    magnitude: float | None = None
    timing: str | None = None


class _ExtractedRemedy(BaseModel):
    action: str
    object: str | None = None
    day: str | None = None
    duration: str | None = None
    prohibition: str | None = None


class _ExtractedRule(BaseModel):
    chunk_index: int
    verbatim_span: str
    facts: list[_ExtractedFact]
    effect: _ExtractedEffect
    remedy: _ExtractedRemedy | None = None
    rule_type: RuleType
    confidence: float


class _RuleExtractionResult(BaseModel):
    rules: list[_ExtractedRule]


class _RuleVerificationOut(BaseModel):
    rule_index: int
    supported: bool
    reason: str


class _RuleVerificationResult(BaseModel):
    verifications: list[_RuleVerificationOut]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _call_parse_with_retry(
    client: Any,
    *,
    model: str,
    system: str,
    user_content: str,
    output_format: type,
    max_tokens: int,
    max_retries: int,
    label: str,
    usage: UsageTracker,
) -> Any:
    """`usage` is reused from `ragcore.embedding.cost` rather than
    duplicated — it's a generic token/request/cost counter with
    nothing embedding-specific about it (same reuse discipline
    `ragcore.claims.claude` already follows for `_format_chunks`).
    Phase 10 §8's "quotas apply to extraction jobs (token cost
    metering)" needs real token counts to meter against; this is
    where they're captured, straight from the API response's own
    `usage` field (exact, billed tokens) rather than a client-side
    estimate."""
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.messages.parse(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_content}],
                output_format=output_format,
            )
            usage.record_success(response.usage.input_tokens + response.usage.output_tokens)
            return response.parsed_output
        except anthropic.RateLimitError as e:
            last_err = e
        except anthropic.APIConnectionError as e:
            last_err = e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                last_err = e
            else:
                usage.record_failure()
                raise GenerationError(f"{model} {label} request rejected: {e}") from e

        if attempt < max_retries - 1:
            time.sleep(min(2**attempt, 30))

    usage.record_failure()
    raise GenerationError(f"{model} {label} failed after {max_retries} attempts: {last_err}") from last_err


def _require_anthropic(client: Any | None, what: str) -> None:
    if client is None and anthropic is None:
        raise GenerationError(
            f"Claude {what} requires the 'generation' extra: install ragcore[generation]"
        ) from _IMPORT_ERROR


class ClaudeRuleCandidateDetector:
    """RuleCandidateDetector implementation using Claude (cheap-tier by
    default — see ragcore.model_routing; a filter, not final-answer
    synthesis)."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_ROUTING.cheap,
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 4096,
        max_retries: int = 5,
    ) -> None:
        _require_anthropic(client, "rule candidate detection")
        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )
        self.usage = UsageTracker()

    def detect(self, chunks: Sequence[Chunk], *, domain_pack: DomainPack) -> list[CandidateSpan]:
        if not chunks:
            return []

        formatted, index_to_chunk_id = _format_chunks(chunks)
        system = (
            f"{domain_pack.extraction_system_prompt()}\n\n"
            "You are given numbered chunks of book text, each labeled [n]. "
            "For each chunk, quote every verbatim span (exact substring, no "
            "paraphrasing) that states a rule of this domain — a conditional "
            "statement linking some chart/situational condition to an "
            "effect or remedy. A chunk may contain zero, one, or several "
            "such spans. Do not extract narrative, mythology, or "
            "explanatory prose that states no actual rule."
        )
        parsed = _call_parse_with_retry(
            self._client, model=self.model, system=system, user_content=formatted,
            output_format=_CandidateDetectionResult, max_tokens=self._max_tokens,
            max_retries=self._max_retries, label="rule candidate detection", usage=self.usage,
        )

        candidates: list[CandidateSpan] = []
        for c in parsed.candidates:
            chunk_id = index_to_chunk_id.get(c.chunk_index)
            if chunk_id is None:
                continue  # hallucinated chunk index
            chunk = next((ch for ch in chunks if ch.chunk_id == chunk_id), None)
            if chunk is None:
                continue
            candidates.append(
                CandidateSpan(chunk_id=chunk_id, source_id=chunk.meta.source_id, text=c.text)
            )
        return candidates


def _build_conditions(
    facts: Sequence[_ExtractedFact], *, domain_pack: DomainPack
) -> Condition | None:
    """Normalizes every extracted fact mention through the domain
    pack's ontology and wraps the result as a Condition. Returns None
    if ANY fact fails to normalize (unknown key, or an unknown value
    for a non-exists operator) — a rule built from a partial condition
    set would silently fire on weaker conditions than the source text
    actually states, which Phase 10's "No span, no rule" (nothing
    unsupported survives) rules out just as much as a missing span
    does."""
    predicates: list[FactPredicate] = []
    for f in facts:
        fact_key = domain_pack.ontology.normalize_fact_key(f.fact_mention)
        if fact_key is None:
            return None
        if f.operator == "exists":
            predicates.append(FactPredicate(fact=fact_key, operator="exists"))
            continue
        if f.value_mention is None:
            return None
        value = domain_pack.ontology.normalize_fact_value(fact_key, f.value_mention)
        if value is None:
            return None
        predicates.append(FactPredicate(fact=fact_key, operator=f.operator, value=value))

    if not predicates:
        return None
    if len(predicates) == 1:
        return predicates[0]
    return ConditionGroup(op="and", children=predicates)


class ClaudeRuleExtractor:
    """RuleExtractor implementation using Claude (cheap-tier by
    default)."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_ROUTING.cheap,
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 8192,
        max_retries: int = 5,
        extraction_version: str = EXTRACTOR_PROMPT_VERSION,
    ) -> None:
        _require_anthropic(client, "rule extraction")
        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._extraction_version = extraction_version
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )
        self.usage = UsageTracker()

    def extract(
        self, candidates: Sequence[CandidateSpan], *, domain_pack: DomainPack
    ) -> list[Rule]:
        if not candidates:
            return []

        numbered = "\n".join(f"[{i}] {c.text}" for i, c in enumerate(candidates, start=1))
        system = (
            f"{domain_pack.extraction_system_prompt()}\n\n"
            "You are given numbered candidate rule spans, each labeled [n]. "
            "For each one that is a genuine rule, extract: verbatim_span "
            "(copy the exact text, do not paraphrase), facts (every "
            "condition it states, as fact_mention/operator/value_mention "
            "using the source's own vocabulary — do not translate to any "
            "canonical form yourself), effect, remedy (if any), rule_type "
            "(HARD if stated as definite, TENDENCY if hedged, EXCEPTION, "
            "CANCELLATION, or REMEDY_ONLY if there's no independent "
            "effect), and your confidence (0-1) that this is a correctly "
            "and completely extracted rule. Skip anything that isn't "
            "actually a rule."
        )
        parsed = _call_parse_with_retry(
            self._client, model=self.model, system=system, user_content=numbered,
            output_format=_RuleExtractionResult, max_tokens=self._max_tokens,
            max_retries=self._max_retries, label="rule extraction", usage=self.usage,
        )

        index_to_candidate = {i: c for i, c in enumerate(candidates, start=1)}
        now = _now()
        rules: list[Rule] = []
        for r in parsed.rules:
            candidate = index_to_candidate.get(r.chunk_index)
            if candidate is None:
                continue  # hallucinated index

            conditions = _build_conditions(r.facts, domain_pack=domain_pack)
            if conditions is None:
                continue  # an unresolvable fact - drop rather than extract a weaker rule

            effect = Effect(
                category=r.effect.category, polarity=r.effect.polarity,
                magnitude=r.effect.magnitude, timing=r.effect.timing,
            )
            remedy = (
                Remedy(
                    action=r.remedy.action, object=r.remedy.object, day=r.remedy.day,
                    duration=r.remedy.duration, prohibition=r.remedy.prohibition,
                )
                if r.remedy is not None
                else None
            )

            rule_id = rule_content_hash(
                domain=domain_pack.domain, book_id=candidate.source_id,
                verbatim_span=r.verbatim_span, conditions=conditions, effect=effect,
            )
            rules.append(
                Rule(
                    rule_id=rule_id,
                    domain=domain_pack.domain,
                    pack_id=domain_pack.pack_id,
                    pack_version=domain_pack.pack_version,
                    book_id=candidate.source_id,
                    page_start=1,
                    chunk_ids=(candidate.chunk_id,),
                    verbatim_span=r.verbatim_span,
                    language="en",
                    normalized_statement=r.verbatim_span,
                    conditions=conditions,
                    effect=effect,
                    remedy=remedy,
                    rule_type=r.rule_type,
                    confidence=max(0.0, min(1.0, r.confidence)),
                    extraction_model=self.model,
                    extraction_version=self._extraction_version,
                    created_at=now,
                )
            )
        return rules


class ClaudeRuleVerifier:
    """RuleVerifier implementation using Claude. Independent of
    ClaudeRuleExtractor by construction — a fresh call given only the
    verbatim span and the already-extracted structured fields, with no
    memory of having proposed them, so it cannot simply agree with its
    own prior output."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_ROUTING.cheap,
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 4096,
        max_retries: int = 5,
    ) -> None:
        _require_anthropic(client, "rule verification")
        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )
        self.usage = UsageTracker()

    def verify(self, rules: Sequence[Rule], *, domain_pack: DomainPack) -> list[VerificationResult]:
        if not rules:
            return []

        domains = {r.domain for r in rules}
        if domains != {domain_pack.domain}:
            raise GenerationError(
                f"ClaudeRuleVerifier.verify() got rules from domain(s) {sorted(domains)} "
                f"but domain_pack.domain={domain_pack.domain!r} - call it once per domain, "
                f"matched to the right pack"
            )

        system = f"{domain_pack.verification_system_prompt()}\n\n{_VERIFIER_SYSTEM_PROMPT_SUFFIX}"
        numbered = "\n\n".join(
            f"[{i}] verbatim_span: {r.verbatim_span!r}\n"
            f"    conditions: {r.conditions.model_dump_json()}\n"
            f"    effect: {r.effect.model_dump_json()}"
            for i, r in enumerate(rules, start=1)
        )
        parsed = _call_parse_with_retry(
            self._client, model=self.model, system=system,
            user_content=numbered, output_format=_RuleVerificationResult,
            max_tokens=self._max_tokens, max_retries=self._max_retries,
            label="rule verification", usage=self.usage,
        )

        index_to_rule = {i: r for i, r in enumerate(rules, start=1)}
        results: list[VerificationResult] = []
        seen: set[str] = set()
        for v in parsed.verifications:
            rule = index_to_rule.get(v.rule_index)
            if rule is None:
                continue
            seen.add(rule.rule_id)
            results.append(VerificationResult(rule_id=rule.rule_id, supported=v.supported, reason=v.reason))

        # Any rule the model didn't address at all is treated as
        # unsupported, not silently passed through - "reject anything
        # unsupported" includes an omitted verdict.
        for rule in rules:
            if rule.rule_id not in seen:
                results.append(
                    VerificationResult(
                        rule_id=rule.rule_id, supported=False,
                        reason="verifier did not return a verdict for this rule",
                    )
                )
        return results


_VERIFIER_SYSTEM_PROMPT_SUFFIX = (
    "You are an independent verifier. You are given numbered candidate "
    "rules, each with its verbatim_span (the exact source text it was "
    "extracted from), its structured conditions (as JSON), and its "
    "structured effect (as JSON). For each one, judge: does the "
    "verbatim_span actually state every one of these conditions and this "
    "effect - not a plausible inference, not something true in general "
    "about this domain, but something this specific span actually says? "
    "If the span supports some but not all of the stated conditions, or "
    "states a different effect, mark it unsupported. Give a one-sentence "
    "reason for every verdict, supported or not."
)
