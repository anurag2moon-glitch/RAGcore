"""Phase 10: rule extraction and rule base, domain-pluggable. See
docs/adr/0013-rule-extraction-domain-pluggable-architecture.md for the
architecture (generic infrastructure here, concrete domains as sibling
packages — e.g. domains/lal_kitab/).

Requires the `generation` extra (`pip install ragcore[generation]`)
for `pydantic` (the rule schema) and `anthropic` (the extraction
pipeline) — the same extra every other structured-LLM-output component
in this codebase already needs.
"""

from ragcore.rules.domain import DomainPack, FactTypeSpec, OntologyNormalizer
from ragcore.rules.evaluator import evaluate_condition, match_rules, validate_facts
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
    "ConditionGroup",
    "DomainPack",
    "Effect",
    "FactPredicate",
    "FactTypeSpec",
    "OntologyNormalizer",
    "Remedy",
    "ReviewStatus",
    "Rule",
    "RuleType",
    "evaluate_condition",
    "match_rules",
    "rule_content_hash",
    "validate_facts",
]
