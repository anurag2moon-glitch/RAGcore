"""Phase 10: rule extraction and rule base, domain-pluggable. See
docs/adr/0013-rule-extraction-domain-pluggable-architecture.md for the
architecture (generic infrastructure here, concrete domains as sibling
packages — e.g. domains/lal_kitab/).

Requires the `generation` extra (`pip install ragcore[generation]`)
for `pydantic` (the rule schema) and `anthropic` (the extraction
pipeline) — the same extra every other structured-LLM-output component
in this codebase already needs.
"""

from ragcore.rules.base import (
    CandidateSpan,
    RuleCandidateDetector,
    RuleExtractor,
    RuleStore,
    RuleVerifier,
    VerificationResult,
)
from ragcore.rules.claude import (
    ClaudeRuleCandidateDetector,
    ClaudeRuleExtractor,
    ClaudeRuleVerifier,
)
from ragcore.rules.consistency import (
    RuleConflict,
    detect_conflicts,
    find_contradictions,
    find_duplicates,
    find_exceptions,
    resolve_precedence,
)
from ragcore.rules.domain import DomainPack, FactTypeSpec, OntologyNormalizer
from ragcore.rules.evaluator import evaluate_condition, match_rules, validate_facts
from ragcore.rules.ocr_quality import is_low_quality_ocr
from ragcore.rules.packs import (
    PackImportError,
    RulePackBundle,
    export_pack,
    import_pack,
)
from ragcore.rules.pipeline import (
    RuleExtractionProgress,
    RuleExtractionReport,
    build_rules,
)
from ragcore.rules.review import ReviewAuditEntry, ReviewQueue
from ragcore.rules.serving import MatchedRule, RuleMatchResult, match_with_provenance
from ragcore.rules.sqlite_store import SqliteRuleStore
from ragcore.rules.types import (
    ConditionGroup,
    Effect,
    FactPredicate,
    Remedy,
    ReviewStatus,
    Rule,
    RuleType,
    rule_content_hash,
)

__all__ = [
    "CandidateSpan",
    "ClaudeRuleCandidateDetector",
    "ClaudeRuleExtractor",
    "ClaudeRuleVerifier",
    "ConditionGroup",
    "DomainPack",
    "Effect",
    "FactPredicate",
    "FactTypeSpec",
    "MatchedRule",
    "OntologyNormalizer",
    "PackImportError",
    "Remedy",
    "ReviewAuditEntry",
    "ReviewQueue",
    "ReviewStatus",
    "Rule",
    "RuleCandidateDetector",
    "RuleConflict",
    "RuleExtractionProgress",
    "RuleExtractionReport",
    "RuleExtractor",
    "RuleMatchResult",
    "RulePackBundle",
    "RuleStore",
    "RuleType",
    "RuleVerifier",
    "SqliteRuleStore",
    "VerificationResult",
    "build_rules",
    "detect_conflicts",
    "evaluate_condition",
    "export_pack",
    "find_contradictions",
    "find_duplicates",
    "find_exceptions",
    "import_pack",
    "is_low_quality_ocr",
    "match_rules",
    "match_with_provenance",
    "resolve_precedence",
    "rule_content_hash",
    "validate_facts",
]
