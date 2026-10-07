"""Rule evaluator (Phase 10 §5): "Evaluator library (Python, pure
functions, no LLM) usable directly by the deterministic engine."

`evaluate_condition` is the single source of truth for a Condition
tree's semantics — `FactPredicate.to_json_logic`/`ConditionGroup.to_json_logic`
(ragcore.rules.types) describe the *same* semantics in JSON Logic form
for an external engine to consume, but this module is what ragcore
itself actually runs facts against, and what match_rules/this module's
own tests hold to be correct.

**Missing-fact semantics, stated once, here**: if `Condition.fact` is
not present in the evaluated fact dict at all, every leaf predicate —
including `ne`/`not_in`, which might look "true" under an open-world
reading of absence — evaluates to `False`. This is a deliberate,
conservative (closed-world) choice: a rule must never fire because
data about it happens to be missing, only because the data that *is*
present genuinely satisfies the condition. `exists` is the one
operator this choice doesn't affect by definition (it's checking
presence itself).
"""

from __future__ import annotations

from typing import Any, Sequence

from ragcore.rules.domain import FactTypeSpec
from ragcore.rules.types import Condition, ConditionGroup, FactPredicate, Rule

FactSet = dict[str, Any]

_MISSING = object()


def _evaluate_predicate(predicate: FactPredicate, facts: FactSet) -> bool:
    if predicate.operator == "exists":
        return predicate.fact in facts

    actual = facts.get(predicate.fact, _MISSING)
    if actual is _MISSING:
        return False

    if predicate.operator == "eq":
        return actual == predicate.value
    if predicate.operator == "ne":
        return actual != predicate.value
    if predicate.operator == "in":
        return actual in predicate.value
    if predicate.operator == "not_in":
        return actual not in predicate.value
    if predicate.operator == "gt":
        return actual > predicate.value
    if predicate.operator == "gte":
        return actual >= predicate.value
    if predicate.operator == "lt":
        return actual < predicate.value
    if predicate.operator == "lte":
        return actual <= predicate.value
    raise AssertionError(f"unhandled operator {predicate.operator!r}")  # pragma: no cover


def _evaluate_group(group: ConditionGroup, facts: FactSet) -> bool:
    if group.op == "not":
        return not evaluate_condition(group.children[0], facts)
    if group.op == "and":
        return all(evaluate_condition(child, facts) for child in group.children)
    if group.op == "or":
        return any(evaluate_condition(child, facts) for child in group.children)
    raise AssertionError(f"unhandled op {group.op!r}")  # pragma: no cover


def evaluate_condition(condition: Condition, facts: FactSet) -> bool:
    """Evaluates a `Condition` tree (a `FactPredicate` leaf or a
    `ConditionGroup` composite) against a flat fact dict. Pure — no
    I/O, no LLM call, safe to call per-rule in a tight loop."""
    if isinstance(condition, FactPredicate):
        return _evaluate_predicate(condition, facts)
    return _evaluate_group(condition, facts)


def validate_facts(facts: FactSet, fact_types: Sequence[FactTypeSpec]) -> list[str]:
    """Returns one human-readable message per fact in `facts` that
    either isn't in `fact_types` at all, or has a value
    `FactTypeSpec.validate_value` rejects — an empty list means every
    fact is recognized and well-typed. Does **not** require every
    known fact type to be present (a fact set is commonly a strict
    subset of a domain's full registry — only the facts relevant to
    one chart/case)."""
    by_key = {ft.fact_key: ft for ft in fact_types}
    errors: list[str] = []
    for key, value in facts.items():
        spec = by_key.get(key)
        if spec is None:
            errors.append(f"unknown fact {key!r} (not in this domain's fact-type registry)")
            continue
        if not spec.validate_value(value):
            errors.append(f"fact {key!r} has value {value!r}, invalid for type {spec.value_type!r}")
    return errors


def match_rules(rules: Sequence[Rule], facts: FactSet) -> list[Rule]:
    """Every rule in `rules` whose `conditions` evaluate `True` against
    `facts`, in input order. Does not itself apply precedence/conflict
    resolution — see `ragcore.rules.consistency` for that; this is
    strictly "which rules' conditions are satisfied," the input that
    layer resolves."""
    return [r for r in rules if evaluate_condition(r.conditions, facts)]
