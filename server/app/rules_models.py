"""Pydantic request/response models for the Phase 10 rule-serving
routes — a projection, not a 1:1 mirror of Rule's own ~20 fields, same
philosophy as app.models's own docstring states for the retrieval API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ragcore.rules.serving import RuleMatchResult
from ragcore.rules.types import Rule


class RuleResponse(BaseModel):
    rule_id: str
    domain: str
    book_id: str
    chapter: str | None
    page_start: int
    verbatim_span: str
    normalized_statement: str
    conditions: dict[str, Any]
    effect: dict[str, Any]
    remedy: dict[str, Any] | None
    rule_type: str
    confidence: float
    review_status: str
    version: int

    @classmethod
    def from_rule(cls, rule: Rule) -> RuleResponse:
        return cls(
            rule_id=rule.rule_id, domain=rule.domain, book_id=rule.book_id,
            chapter=rule.chapter, page_start=rule.page_start,
            verbatim_span=rule.verbatim_span, normalized_statement=rule.normalized_statement,
            conditions=rule.conditions.model_dump(), effect=rule.effect.model_dump(),
            remedy=rule.remedy.model_dump() if rule.remedy else None,
            rule_type=rule.rule_type, confidence=rule.confidence,
            review_status=rule.review_status, version=rule.version,
        )


class MatchRulesRequest(BaseModel):
    domain: str
    facts: dict[str, Any]
    book_id: str | None = None
    include_unverified: bool = False


class MatchedRuleResponse(BaseModel):
    rule: RuleResponse
    is_overridden: bool
    overrides: tuple[str, ...]


class MatchRulesResponse(BaseModel):
    matched: list[MatchedRuleResponse]
    effective: list[RuleResponse]

    @classmethod
    def from_result(cls, result: RuleMatchResult) -> MatchRulesResponse:
        return cls(
            matched=[
                MatchedRuleResponse(
                    rule=RuleResponse.from_rule(m.rule), is_overridden=m.is_overridden,
                    overrides=m.overrides,
                )
                for m in result.matched
            ],
            effective=[RuleResponse.from_rule(r) for r in result.effective],
        )


class ReviewActionRequest(BaseModel):
    reviewer: str
    reason: str = ""


class EditRuleRequest(BaseModel):
    reviewer: str
    reason: str
    effect: dict[str, Any] | None = None
    confidence: float | None = None
    remedy: dict[str, Any] | None = None


class ExtractRulesResponse(BaseModel):
    chapters_processed: int
    chapters_skipped: int
    candidates_detected: int
    rules_extracted: int
    rules_stored: int
    rules_rejected_unsupported: int
    errors: list[tuple[str, str, str]]
    total_tokens_used: int
