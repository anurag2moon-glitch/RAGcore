"""ragcore.rules.types: the rule schema's validators, JSON Logic
compilation, and content-addressed rule_content_hash — Phase 10 §1.

JSON Logic compilation is tested behaviorally, against a *real* JSON
Logic engine (`json-logic-qubit`, imported as `json_logic`) rather than
by asserting an exact compiled shape — a brittle test that would break
on any internal recompilation change without catching an actual
semantic bug, which is exactly what matters here. This found two real
bugs during development, not hypothetically: `{"!!": {"var": fact}}`
for `exists` is a *truthiness* check, not a *presence* check (wrongly
`False` when the fact is present but falsy — `0`, `False`, `""`), and
a bare comparison operator against a missing fact followed json-logic's
own null-propagation rules instead of ragcore's deliberately
conservative "missing fact -> predicate is False" semantics (see
`FactPredicate.to_json_logic`'s own docstring for the fix). Both are
regression-tested here via the real engine, not just asserted fixed.
"""

from __future__ import annotations

import random

import pytest
from json_logic import jsonLogic
from pydantic import ValidationError

from ragcore.rules.evaluator import evaluate_condition
from ragcore.rules.types import (
    Condition,
    ConditionGroup,
    Effect,
    FactPredicate,
    Remedy,
    Rule,
    rule_content_hash,
)


def _cond() -> ConditionGroup:
    return ConditionGroup(
        op="and",
        children=[
            FactPredicate(fact="planet.mars.house", operator="eq", value=8),
            FactPredicate(fact="planet.mars.retrograde", operator="exists"),
        ],
    )


def _effect() -> Effect:
    return Effect(category="health", polarity="negative", magnitude=0.7, timing="during Mars dasha")


def _rule_kwargs(**overrides) -> dict:
    base = {
        "rule_id": "rule:abc123",
        "domain": "lal_kitab",
        "pack_id": "lal_kitab_v1",
        "pack_version": "1.0.0",
        "book_id": "book1",
        "chapter": "3",
        "page_start": 42,
        "chunk_ids": ("book1:5",),
        "verbatim_span": "Mars in the 8th house causes health troubles.",
        "language": "en",
        "normalized_statement": "Mars in house 8 -> health troubles",
        "conditions": _cond(),
        "effect": _effect(),
        "rule_type": "HARD",
        "confidence": 0.9,
        "extraction_model": "claude-opus-5",
        "extraction_version": "v1",
        "created_at": "2026-10-07T00:00:00Z",
    }
    base.update(overrides)
    return base


# -- FactPredicate ----------------------------------------------------------


def test_fact_predicate_rejects_an_empty_fact() -> None:
    with pytest.raises(ValidationError):
        FactPredicate(fact="", operator="eq", value=1)


def test_fact_predicate_requires_a_value_unless_exists() -> None:
    with pytest.raises(ValidationError):
        FactPredicate(fact="x", operator="eq")


def test_fact_predicate_exists_needs_no_value() -> None:
    pred = FactPredicate(fact="x", operator="exists")
    assert pred.value is None


@pytest.mark.parametrize("operator", ["eq", "ne", "gt", "gte", "lt", "lte"])
def test_fact_predicate_json_logic_comparison_matches_real_engine_when_present(operator: str) -> None:
    pred = FactPredicate(fact="x", operator=operator, value=5)
    for x in (3, 5, 7):
        facts = {"x": x}
        assert jsonLogic(pred.to_json_logic(), facts) == evaluate_condition(pred, facts)


@pytest.mark.parametrize("operator", ["eq", "ne", "gt", "gte", "lt", "lte"])
def test_fact_predicate_json_logic_comparison_is_false_when_missing(operator: str) -> None:
    pred = FactPredicate(fact="x", operator=operator, value=5)
    assert jsonLogic(pred.to_json_logic(), {}) is False
    assert evaluate_condition(pred, {}) is False


def test_fact_predicate_json_logic_in_matches_real_engine() -> None:
    pred = FactPredicate(fact="x", operator="in", value=[1, 2, 3])
    assert jsonLogic(pred.to_json_logic(), {"x": 2}) == evaluate_condition(pred, {"x": 2})
    assert jsonLogic(pred.to_json_logic(), {"x": 9}) == evaluate_condition(pred, {"x": 9})
    assert jsonLogic(pred.to_json_logic(), {}) is False


def test_fact_predicate_json_logic_not_in_matches_real_engine() -> None:
    pred = FactPredicate(fact="x", operator="not_in", value=[1, 2])
    assert jsonLogic(pred.to_json_logic(), {"x": 9}) == evaluate_condition(pred, {"x": 9})
    assert jsonLogic(pred.to_json_logic(), {"x": 1}) == evaluate_condition(pred, {"x": 1})
    assert jsonLogic(pred.to_json_logic(), {}) is False


def test_fact_predicate_json_logic_exists_is_a_presence_check_not_a_truthiness_check() -> None:
    """The specific bug found during development: `{"!!": {"var": x}}`
    (truthiness) wrongly says "doesn't exist" for a present-but-falsy
    fact. The fix must say the fact exists for every one of these."""
    pred = FactPredicate(fact="x", operator="exists")
    for falsy_value in (0, False, ""):
        facts = {"x": falsy_value}
        assert jsonLogic(pred.to_json_logic(), facts) is True
        assert evaluate_condition(pred, facts) is True
    assert jsonLogic(pred.to_json_logic(), {}) is False
    assert evaluate_condition(pred, {}) is False


# -- ConditionGroup -----------------------------------------------------


def test_condition_group_not_requires_exactly_one_child() -> None:
    with pytest.raises(ValidationError):
        ConditionGroup(
            op="not",
            children=[
                FactPredicate(fact="a", operator="exists"),
                FactPredicate(fact="b", operator="exists"),
            ],
        )


def test_condition_group_and_requires_at_least_one_child() -> None:
    with pytest.raises(ValidationError):
        ConditionGroup(op="and", children=[])


def test_condition_group_json_logic_and_matches_real_engine() -> None:
    group = ConditionGroup(
        op="and",
        children=[FactPredicate(fact="a", operator="exists"), FactPredicate(fact="b", operator="exists")],
    )
    for facts in ({"a": 1, "b": 1}, {"a": 1}, {}):
        assert jsonLogic(group.to_json_logic(), facts) == evaluate_condition(group, facts)


def test_condition_group_json_logic_not_matches_real_engine() -> None:
    group = ConditionGroup(op="not", children=[FactPredicate(fact="a", operator="eq", value=1)])
    for facts in ({"a": 1}, {"a": 2}, {}):
        assert jsonLogic(group.to_json_logic(), facts) == evaluate_condition(group, facts)


def test_condition_group_json_logic_is_recursive_and_matches_real_engine() -> None:
    nested = ConditionGroup(
        op="or",
        children=[
            ConditionGroup(op="and", children=[FactPredicate(fact="a", operator="exists")]),
            FactPredicate(fact="b", operator="eq", value=1),
        ],
    )
    for facts in ({"a": 1}, {"b": 1}, {}, {"a": 0}):
        assert jsonLogic(nested.to_json_logic(), facts) == evaluate_condition(nested, facts)


def test_condition_round_trips_through_json() -> None:
    """The discriminated union (FactPredicate | ConditionGroup) parses
    back correctly from its own serialized form — exercises the `kind`
    discriminator actually working, not just construction."""
    group = _cond()
    parsed = ConditionGroup.model_validate_json(group.model_dump_json())
    assert parsed.to_json_logic() == group.to_json_logic()


def test_to_json_logic_matches_evaluate_condition_randomized() -> None:
    """A randomized cross-check between the compiled JSON Logic output
    (run through a real JSON Logic engine) and
    ragcore.rules.evaluator.evaluate_condition, across random condition
    trees and fact sets including falsy values (0, False) — the exact
    methodology that found both bugs this module's docstring describes,
    kept as a permanent regression test rather than a one-off
    development check."""
    rng = random.Random(42)

    def random_predicate() -> FactPredicate:
        op = rng.choice(["eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "exists"])
        fact = rng.choice(["a", "b", "c"])
        if op == "exists":
            return FactPredicate(fact=fact, operator=op)
        if op in ("in", "not_in"):
            return FactPredicate(fact=fact, operator=op, value=rng.sample(range(10), 3))
        return FactPredicate(fact=fact, operator=op, value=rng.randint(0, 10))

    def random_condition(depth: int = 0) -> Condition:
        if depth >= 2 or rng.random() < 0.4:
            return random_predicate()
        op = rng.choice(["and", "or", "not"])
        n = 1 if op == "not" else rng.choice([2, 3])
        return ConditionGroup(op=op, children=[random_condition(depth + 1) for _ in range(n)])

    def random_facts() -> dict:
        facts = {}
        for k in ["a", "b", "c"]:
            if rng.random() < 0.8:
                facts[k] = rng.choice([0, rng.randint(0, 10), False])
        return facts

    for _ in range(1000):
        cond = random_condition()
        facts = random_facts()
        assert evaluate_condition(cond, facts) == bool(jsonLogic(cond.to_json_logic(), facts)), (
            f"mismatch for {cond.model_dump()} against {facts}"
        )


# -- Effect / Remedy ------------------------------------------------------


def test_effect_rejects_an_empty_category() -> None:
    with pytest.raises(ValidationError):
        Effect(category="", polarity="positive")


def test_remedy_rejects_an_empty_action() -> None:
    with pytest.raises(ValidationError):
        Remedy(action="")


def test_remedy_optional_fields_default_to_none() -> None:
    remedy = Remedy(action="donate")
    assert remedy.object is None
    assert remedy.day is None


# -- Rule: "no span, no rule" ---------------------------------------------


def test_rule_rejects_page_start_below_one() -> None:
    with pytest.raises(ValidationError):
        Rule(**_rule_kwargs(page_start=0))


def test_rule_rejects_empty_chunk_ids() -> None:
    with pytest.raises(ValidationError):
        Rule(**_rule_kwargs(chunk_ids=()))


def test_rule_rejects_an_empty_verbatim_span() -> None:
    with pytest.raises(ValidationError):
        Rule(**_rule_kwargs(verbatim_span="   "))


def test_rule_rejects_confidence_outside_unit_range() -> None:
    with pytest.raises(ValidationError):
        Rule(**_rule_kwargs(confidence=1.5))


# -- Rule: tenant scope -----------------------------------------------------


def test_rule_tenant_scope_requires_tenant_id() -> None:
    with pytest.raises(ValidationError):
        Rule(**_rule_kwargs(tenant_scope="tenant"))


def test_rule_global_scope_rejects_a_tenant_id() -> None:
    with pytest.raises(ValidationError):
        Rule(**_rule_kwargs(tenant_id="tenant-a"))


def test_rule_tenant_scope_with_tenant_id_is_valid() -> None:
    rule = Rule(**_rule_kwargs(tenant_scope="tenant", tenant_id="tenant-a"))
    assert rule.tenant_id == "tenant-a"


# -- Rule: version/supersedes ------------------------------------------------


def test_rule_version_two_requires_supersedes() -> None:
    with pytest.raises(ValidationError):
        Rule(**_rule_kwargs(version=2))


def test_rule_version_one_rejects_supersedes() -> None:
    with pytest.raises(ValidationError):
        Rule(**_rule_kwargs(supersedes="rule:old"))


def test_rule_version_two_with_supersedes_is_valid() -> None:
    rule = Rule(**_rule_kwargs(version=2, supersedes="rule:old", rule_id="rule:new"))
    assert rule.version == 2
    assert rule.supersedes == "rule:old"


# -- rule_content_hash --------------------------------------------------


def test_rule_content_hash_is_deterministic() -> None:
    h1 = rule_content_hash(domain="d", book_id="b", verbatim_span="text", conditions=_cond(), effect=_effect())
    h2 = rule_content_hash(domain="d", book_id="b", verbatim_span="text", conditions=_cond(), effect=_effect())
    assert h1 == h2
    assert h1.startswith("rule:")


def test_rule_content_hash_differs_on_different_conditions() -> None:
    other_cond = FactPredicate(fact="x", operator="exists")
    h1 = rule_content_hash(domain="d", book_id="b", verbatim_span="text", conditions=_cond(), effect=_effect())
    h2 = rule_content_hash(domain="d", book_id="b", verbatim_span="text", conditions=other_cond, effect=_effect())
    assert h1 != h2


def test_rule_content_hash_differs_on_different_effect() -> None:
    other_effect = Effect(category="wealth", polarity="positive")
    h1 = rule_content_hash(domain="d", book_id="b", verbatim_span="text", conditions=_cond(), effect=_effect())
    h2 = rule_content_hash(domain="d", book_id="b", verbatim_span="text", conditions=_cond(), effect=other_effect)
    assert h1 != h2


def test_rule_content_hash_is_insensitive_to_whitespace_differences_in_span() -> None:
    h1 = rule_content_hash(domain="d", book_id="b", verbatim_span="a  b\tc", conditions=_cond(), effect=_effect())
    h2 = rule_content_hash(domain="d", book_id="b", verbatim_span="a b c", conditions=_cond(), effect=_effect())
    assert h1 == h2


def test_rule_content_hash_differs_on_different_book_id() -> None:
    h1 = rule_content_hash(domain="d", book_id="book1", verbatim_span="text", conditions=_cond(), effect=_effect())
    h2 = rule_content_hash(domain="d", book_id="book2", verbatim_span="text", conditions=_cond(), effect=_effect())
    assert h1 != h2


# -- Rule.to_json_logic / full construction ---------------------------------


def test_rule_to_json_logic_delegates_to_its_conditions() -> None:
    rule = Rule(**_rule_kwargs())
    assert rule.to_json_logic() == rule.conditions.to_json_logic()


def test_rule_full_construction_and_json_round_trip() -> None:
    rule = Rule(**_rule_kwargs())
    parsed = Rule.model_validate_json(rule.model_dump_json())
    assert parsed.rule_id == rule.rule_id
    assert parsed.to_json_logic() == rule.to_json_logic()


def test_rule_json_schema_generation_does_not_raise() -> None:
    schema = Rule.model_json_schema()
    assert schema["title"] == "Rule"
    assert "rule_id" in schema["required"]
    assert "page_start" in schema["required"]
    assert "chunk_ids" in schema["required"]
