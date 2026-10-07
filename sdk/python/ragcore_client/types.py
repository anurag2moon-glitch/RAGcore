"""Response types for ragcore_client.

Plain dataclasses, not a reuse of ragcore.types — this SDK talks to the
reference server's HTTP JSON contract (server/app/models.py), not to
ragcore's own in-process dataclasses, and a real third-party consumer of
this SDK never imports ragcore directly at all.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IngestResult:
    source_ids: list[str]
    chunks_ingested: int
    sources_replaced: list[str]
    errors: list[tuple[str, str]]


@dataclass(frozen=True)
class SourceInfo:
    source_id: str
    path: str
    title: str | None
    source_type: str
    chunk_count: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source_id: str
    source_path: str
    chapter_title: str | None
    tenant_id: str | None
    license_id: str | None


@dataclass(frozen=True)
class AskResult:
    text: str
    citations: list[RetrievedChunk]


@dataclass(frozen=True)
class RuleInfo:
    """Mirrors server/app/rules_models.py's RuleResponse — a
    projection of ragcore.rules.types.Rule, not a 1:1 mirror, same
    philosophy as every other type in this file."""

    rule_id: str
    domain: str
    book_id: str
    chapter: str | None
    page_start: int
    verbatim_span: str
    normalized_statement: str
    conditions: dict[str, object]
    effect: dict[str, object]
    remedy: dict[str, object] | None
    rule_type: str
    confidence: float
    review_status: str
    version: int


@dataclass(frozen=True)
class MatchedRuleInfo:
    rule: RuleInfo
    is_overridden: bool
    overrides: tuple[str, ...]


@dataclass(frozen=True)
class MatchRulesResult:
    matched: list[MatchedRuleInfo]
    effective: list[RuleInfo]


@dataclass(frozen=True)
class ExtractRulesResult:
    chapters_processed: int
    chapters_skipped: int
    candidates_detected: int
    rules_extracted: int
    rules_stored: int
    rules_rejected_unsupported: int
    errors: list[tuple[str, str, str]]
    total_tokens_used: int
