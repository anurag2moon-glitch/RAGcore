"""Rule schema (Phase 10 §1) — versioned, Pydantic + JSON Schema, per
the spec. Requires the `generation` extra (`pip install ragcore[generation]`)
for `pydantic` — the same extra every other structured-LLM-output
component in this codebase already needs, not a new dependency surface
for this layer specifically.

Domain-agnostic by design (docs/adr/0013-rule-extraction-domain-pluggable-architecture.md):
nothing here mentions Lal Kitab, planets, or houses. `Condition.fact`
is an open-vocabulary string a domain pack defines the real values
for; this schema only enforces the *shape* (a predicate tree over
typed facts), never the vocabulary.
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "1"

RuleType = Literal["HARD", "TENDENCY", "EXCEPTION", "CANCELLATION", "REMEDY_ONLY"]
"""HARD: stated as definite. TENDENCY: hedged ("may", "tends to").
EXCEPTION: carves out a case another rule's conditions would otherwise
cover. CANCELLATION: states that another rule's effect is negated
under additional conditions. REMEDY_ONLY: no independent effect, just
a prescribed remedy for a condition already established elsewhere."""

ReviewStatus = Literal["auto", "verified", "rejected"]
"""auto: extracted, not yet reviewed by a human — the only status
ragcore.rules itself ever assigns. verified/rejected: set exclusively
by the review workflow (ragcore.rules.review), never by extraction."""

ConditionOperator = Literal["eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "exists"]
"""Comparison operators a FactPredicate leaf evaluates `fact` by.
`exists` ignores `value` (checks only that `fact` is present in the
evaluated fact set) — every other operator requires `value`."""


class FactPredicate(BaseModel):
    """A leaf condition: one fact compared against a value. `fact` is
    an open-vocabulary dotted key (e.g. "planet.mars.house") whose
    real vocabulary a domain pack's ontology defines — this schema
    enforces only that it's a non-empty string, never a specific
    taxonomy, so a new domain never needs a core schema change to
    introduce its own fact keys."""

    kind: Literal["predicate"] = "predicate"
    fact: str
    operator: ConditionOperator
    value: Any = None

    @field_validator("fact")
    @classmethod
    def _fact_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("fact must be a non-empty string")
        return v

    @model_validator(mode="after")
    def _value_required_unless_exists(self) -> FactPredicate:
        if self.operator != "exists" and self.value is None:
            raise ValueError(f"operator {self.operator!r} requires a non-None value")
        return self

    def to_json_logic(self) -> dict[str, Any]:
        """Compiles to a JSON-Logic-style (https://jsonlogic.com)
        expression — Phase 10 §5's "compile conditions to a JSON-logic-
        style DSL." `exists` has no native JSON Logic operator, so it
        compiles to `{"!!": {"var": fact}}` (truthy-coerced presence
        check), the idiomatic JSON Logic way to express "this var is
        set and truthy"."""
        if self.operator == "exists":
            return {"!!": {"var": self.fact}}
        op_map = {
            "eq": "==", "ne": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<=",
        }
        if self.operator in op_map:
            return {op_map[self.operator]: [{"var": self.fact}, self.value]}
        if self.operator == "in":
            return {"in": [{"var": self.fact}, self.value]}
        if self.operator == "not_in":
            return {"!": {"in": [{"var": self.fact}, self.value]}}
        raise AssertionError(f"unhandled operator {self.operator!r}")  # pragma: no cover


class ConditionGroup(BaseModel):
    """A composite condition: AND/OR over 1+ children, or NOT over
    exactly 1. Recursive — `children` holds further `Condition`s
    (predicates or nested groups)."""

    kind: Literal["group"] = "group"
    op: Literal["and", "or", "not"]
    children: list[Condition]

    @model_validator(mode="after")
    def _children_count_matches_op(self) -> ConditionGroup:
        if self.op == "not" and len(self.children) != 1:
            raise ValueError(f"'not' requires exactly 1 child, got {len(self.children)}")
        if self.op in ("and", "or") and len(self.children) < 1:
            raise ValueError(f"{self.op!r} requires at least 1 child")
        return self

    def to_json_logic(self) -> dict[str, Any]:
        compiled = [c.to_json_logic() for c in self.children]
        if self.op == "not":
            return {"!": compiled[0]}
        return {self.op: compiled}


Condition = Annotated[FactPredicate | ConditionGroup, Field(discriminator="kind")]
ConditionGroup.model_rebuild()


class Effect(BaseModel):
    category: str
    """Open vocabulary (e.g. "health", "wealth", "marriage", "career",
    "debt") — a domain pack's own taxonomy, not enumerated here."""
    polarity: Literal["positive", "negative", "neutral"]
    magnitude: float | None = None
    """Only set when the source text states a strength/degree — never
    invented to fill the field."""
    timing: str | None = None
    """Free text when the source states a timing qualifier (e.g.
    "during Mars dasha") — not a structured fact, since timing
    vocabulary varies too much across domains to standardize here."""

    @field_validator("category")
    @classmethod
    def _category_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("category must be a non-empty string")
        return v


class Remedy(BaseModel):
    action: str
    object: str | None = None
    day: str | None = None
    duration: str | None = None
    prohibition: str | None = None

    @field_validator("action")
    @classmethod
    def _action_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("action must be a non-empty string")
        return v


class Rule(BaseModel):
    """One extracted rule. `rule_id` is content-addressed —
    `rule_content_hash(domain, book_id, verbatim_span, conditions,
    effect)` below — so re-extracting unchanged source text reproduces
    the same id (docs/adr/0010-content-addressed-identity.md's scheme,
    applied here).

    **"No span, no rule"**: `verbatim_span`, `page_start`, and
    `chunk_ids` are all required and non-empty — enforced here, not
    left to extraction-time discipline, so a rule that somehow lost
    its provenance can never be constructed, let alone stored.
    """

    rule_id: str
    domain: str
    pack_id: str
    pack_version: str
    book_id: str
    chapter: str | None = None
    page_start: int
    page_end: int | None = None
    chunk_ids: tuple[str, ...]
    verbatim_span: str
    language: str
    normalized_statement: str

    conditions: Condition
    effect: Effect
    remedy: Remedy | None = None

    rule_type: RuleType
    depends_on: tuple[str, ...] = ()
    overrides: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()

    confidence: float
    extraction_model: str
    extraction_version: str
    review_status: ReviewStatus = "auto"

    tenant_scope: Literal["global", "tenant"] = "global"
    tenant_id: str | None = None

    version: int = 1
    """Incremented only by the review workflow's edit path
    (`ragcore.rules.review`) — extraction always produces version 1.
    An edit creates a *new* `Rule` with `version = old.version + 1`
    and a new `rule_id`; it never mutates an existing row in place
    (Phase 10 §4, docs/adr/0013)."""
    supersedes: str | None = None
    """The `rule_id` of the version this one replaces, when
    `version > 1`. `None` for every version-1 rule."""

    created_at: str
    """ISO-8601 UTC timestamp string — display/audit only, never used
    to decide freshness (content-addressing does that)."""

    @field_validator("page_start")
    @classmethod
    def _page_start_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("page_start must be >= 1")
        return v

    @field_validator("chunk_ids")
    @classmethod
    def _chunk_ids_non_empty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("chunk_ids must be non-empty — every rule must trace to real chunks")
        return v

    @field_validator("verbatim_span")
    @classmethod
    def _verbatim_span_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("verbatim_span must be a non-empty string — \"no span, no rule\"")
        return v

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")
        return v

    @model_validator(mode="after")
    def _tenant_id_matches_scope(self) -> Rule:
        if self.tenant_scope == "tenant" and self.tenant_id is None:
            raise ValueError("tenant_scope='tenant' requires tenant_id to be set")
        if self.tenant_scope == "global" and self.tenant_id is not None:
            raise ValueError("tenant_scope='global' must not carry a tenant_id")
        return self

    @model_validator(mode="after")
    def _version_supersedes_consistency(self) -> Rule:
        if self.version == 1 and self.supersedes is not None:
            raise ValueError("version 1 must not set supersedes")
        if self.version > 1 and self.supersedes is None:
            raise ValueError(f"version {self.version} must set supersedes")
        return self

    def to_json_logic(self) -> dict[str, Any]:
        return self.conditions.to_json_logic()


def rule_content_hash(
    *,
    domain: str,
    book_id: str,
    verbatim_span: str,
    conditions: Condition,
    effect: Effect,
) -> str:
    """`f"rule:{sha256(domain|book_id|normalized_span|conditions_json|effect_json)[:16]}"`
    — content-addressed over exactly the fields that define "the same
    rule restated," mirroring `ragcore.claims.claude._claim_id`'s
    scheme (docs/adr/0010). Deliberately excludes `chapter`/`page_*`/
    `chunk_ids`: the same rule stated identically on two different
    pages (a cross-reference, a summary table) should collapse to one
    id, the same reasoning `ragcore.graph`'s content-addressed entity
    ids use for cross-chapter recurrence. `conditions`/`effect` are
    included via Pydantic's own `model_dump_json` — deterministic
    because it serializes in the model's fixed field-definition order,
    not Python dict insertion order, so two structurally-identical
    trees hash identically regardless of how they were constructed."""
    normalized_span = " ".join(verbatim_span.split())
    payload = "|".join(
        [
            domain,
            book_id,
            normalized_span,
            conditions.model_dump_json(exclude_none=True),
            effect.model_dump_json(exclude_none=True),
        ]
    )
    return f"rule:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"
