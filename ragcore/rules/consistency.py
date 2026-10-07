"""Consistency layer (Phase 10 §3): dedup near-identical rules, detect
contradictions and exceptions, store every finding as a `RuleConflict`
record — "never silently overwrite" — and resolve precedence
(specific over general, exception over rule) where that's actually
decidable.

Pure functions over already-extracted `Rule`s, no LLM calls: every
comparison here is over `Condition`/`Effect`, already-structured data
the extraction layer produced, not raw text — this is exactly the kind
of thing `ragcore.rules.evaluator` already established the pattern
for (deterministic, inspectable logic over the schema, not another
model call asked to "judge" consistency).

**Flat-conjunction limitation, stated plainly, not hidden**:
contradiction and exception detection both need to compare two rules'
*condition sets* for overlap/subsumption. That's decidable in general
only for a flat conjunction of facts (a bare `FactPredicate`, or an
`and`-only `ConditionGroup` of `FactPredicate`s) — general subsumption
between two arbitrary `AND`/`OR`/`NOT` trees is a SAT-equivalent
problem, out of scope here. A rule whose conditions use `or`/`not`
anywhere is simply not comparable by `find_contradictions`/
`find_exceptions` (returns `None` from `_flat_facts`, skipped) — it
still participates in `find_duplicates`, which only needs exact
structural equality, not subsumption.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations
from typing import Literal, Sequence

from ragcore.rules.types import Condition, ConditionGroup, FactPredicate, Rule

ConflictType = Literal["duplicate", "contradiction", "exception"]

FactTuple = tuple[str, str, object]
"""(fact, operator, hashable_value) — one leaf predicate, normalized
to a hashable form (a list `value`, e.g. for `in`/`not_in`, becomes a
tuple) so a flat condition set can be a `frozenset`."""


@dataclass(frozen=True)
class RuleConflict:
    conflict_id: str
    conflict_type: ConflictType
    rule_ids: tuple[str, str]
    """Sorted for duplicate/contradiction (symmetric). For exception,
    (general_rule_id, exception_rule_id) — directional, since an
    exception's relationship to the rule it carves from isn't
    symmetric."""
    rationale: str
    winning_rule_id: str | None
    """The rule_id that takes precedence, or None when this
    conflict_type/situation doesn't resolve automatically (a duplicate
    has no "winner" at all; a contradiction with equally-specific
    conditions on both sides is genuinely ambiguous and needs a human
    reviewer, per Phase 10 §4)."""


def _conflict_id(conflict_type: ConflictType, rule_a: str, rule_b: str) -> str:
    payload = f"{conflict_type}:{rule_a}:{rule_b}"
    return f"ruleconflict:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _hashable_value(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


def _flat_facts(condition: Condition) -> frozenset[FactTuple] | None:
    """The condition's facts as a frozenset, if and only if it's a
    flat conjunction (a bare predicate, or an and-only group of
    predicates) — None for anything containing or/not or a nested
    group, which this layer can't safely compare."""
    if isinstance(condition, FactPredicate):
        return frozenset({(condition.fact, condition.operator, _hashable_value(condition.value))})
    if isinstance(condition, ConditionGroup):
        if condition.op != "and":
            return None
        facts: set[FactTuple] = set()
        for child in condition.children:
            if not isinstance(child, FactPredicate):
                return None  # nested group - not flat
            facts.add((child.fact, child.operator, _hashable_value(child.value)))
        return frozenset(facts)
    return None  # pragma: no cover - exhaustive over the Condition union


def find_duplicates(rules: Sequence[Rule]) -> list[RuleConflict]:
    """Rules with structurally identical conditions AND effect within
    the same domain — the same rule restated across pages/books
    (Phase 10 §3's "Dedupe near-identical rules across pages and
    books"). Exact structural equality, not fuzzy text matching: two
    rules whose facts were independently normalized to the same
    canonical ontology terms by extraction are "near-identical" in
    every way that matters to the evaluator, regardless of how
    differently their source prose reads."""
    by_key: dict[tuple[str, str, str], list[Rule]] = {}
    for rule in rules:
        key = (rule.domain, rule.conditions.model_dump_json(), rule.effect.model_dump_json())
        by_key.setdefault(key, []).append(rule)

    conflicts: list[RuleConflict] = []
    for group in by_key.values():
        if len(group) < 2:
            continue
        for a, b in combinations(sorted(group, key=lambda r: r.rule_id), 2):
            conflicts.append(
                RuleConflict(
                    conflict_id=_conflict_id("duplicate", a.rule_id, b.rule_id),
                    conflict_type="duplicate",
                    rule_ids=(a.rule_id, b.rule_id),
                    rationale="identical conditions and effect within the same domain",
                    winning_rule_id=None,
                )
            )
    return conflicts


def find_contradictions(rules: Sequence[Rule]) -> list[RuleConflict]:
    """Same flat condition set, same effect category, opposite
    polarity — Phase 10 §3's "Detect contradictions (same conditions,
    opposite effects)". Resolved via specific-over-general when the
    two rules' condition sets differ in size (more facts = more
    specific); left unresolved (winning_rule_id=None) when they're
    equally specific, since that's a genuine tie this layer has no
    basis to break."""
    by_key: dict[tuple[str, frozenset[FactTuple]], list[Rule]] = {}
    for rule in rules:
        facts = _flat_facts(rule.conditions)
        if facts is None:
            continue
        by_key.setdefault((rule.domain, facts), []).append(rule)

    conflicts: list[RuleConflict] = []
    for group in by_key.values():
        if len(group) < 2:
            continue
        for a, b in combinations(sorted(group, key=lambda r: r.rule_id), 2):
            if a.effect.category != b.effect.category or a.effect.polarity == b.effect.polarity:
                continue
            conflicts.append(
                RuleConflict(
                    conflict_id=_conflict_id("contradiction", a.rule_id, b.rule_id),
                    conflict_type="contradiction",
                    rule_ids=(a.rule_id, b.rule_id),
                    rationale=(
                        f"identical conditions, same effect category {a.effect.category!r}, "
                        f"opposite polarity ({a.effect.polarity!r} vs {b.effect.polarity!r})"
                    ),
                    winning_rule_id=None,  # same flat fact set -> equally specific, no tiebreak
                )
            )
    return conflicts


def find_exceptions(rules: Sequence[Rule]) -> list[RuleConflict]:
    """A rule typed EXCEPTION or CANCELLATION whose flat condition set
    is a proper superset of another rule's — Phase 10 §3's "exceptions"
    and the "exception over rule" precedence principle, resolved at
    detection time since it's definitional, not situational: an
    EXCEPTION/CANCELLATION rule's entire purpose is to override the
    more general rule it carves a case out of, so there's no tie to
    break."""
    conflicts: list[RuleConflict] = []
    by_domain: dict[str, list[Rule]] = {}
    for rule in rules:
        by_domain.setdefault(rule.domain, []).append(rule)

    for domain_rules in by_domain.values():
        flat = [(r, _flat_facts(r.conditions)) for r in domain_rules]
        for general, general_facts in flat:
            if general_facts is None:
                continue
            for specific, specific_facts in flat:
                if specific.rule_id == general.rule_id or specific_facts is None:
                    continue
                if specific.rule_type not in ("EXCEPTION", "CANCELLATION"):
                    continue
                if not (general_facts < specific_facts):  # proper superset
                    continue
                conflicts.append(
                    RuleConflict(
                        conflict_id=_conflict_id("exception", general.rule_id, specific.rule_id),
                        conflict_type="exception",
                        rule_ids=(general.rule_id, specific.rule_id),
                        rationale=(
                            f"{specific.rule_type} rule {specific.rule_id!r}'s conditions are a "
                            f"strict superset of {general.rule_id!r}'s - carves out a more "
                            f"specific case"
                        ),
                        winning_rule_id=specific.rule_id,
                    )
                )
    return conflicts


def detect_conflicts(rules: Sequence[Rule]) -> list[RuleConflict]:
    return find_duplicates(rules) + find_contradictions(rules) + find_exceptions(rules)


def resolve_precedence(conflict: RuleConflict, rules_by_id: dict[str, Rule]) -> str | None:
    """Re-derives the precedence outcome for a conflict already found
    by detect_conflicts — exposed separately so a consumer re-checking
    a stored RuleConflict doesn't need to re-run detection from
    scratch. Duplicate conflicts never resolve (no "winner" concept);
    exception conflicts already carry their resolution at detection
    time; contradiction conflicts resolve via specific-over-general
    when decidable."""
    if conflict.conflict_type != "contradiction":
        return conflict.winning_rule_id

    sizes = {
        rule_id: len(_flat_facts(rules_by_id[rule_id].conditions) or frozenset())
        for rule_id in conflict.rule_ids
    }
    best = max(sizes.values())
    winners = [rule_id for rule_id, size in sizes.items() if size == best]
    return winners[0] if len(winners) == 1 else None
