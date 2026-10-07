"""ragcore.rules.types: the rule schema's validators, JSON Logic
compilation, and content-addressed rule_content_hash — Phase 10 §1.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragcore.rules.types import (
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


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("eq", {"==": [{"var": "x"}, 1]}),
        ("ne", {"!=": [{"var": "x"}, 1]}),
        ("gt", {">": [{"var": "x"}, 1]}),
        ("gte", {">=": [{"var": "x"}, 1]}),
        ("lt", {"<": [{"var": "x"}, 1]}),
        ("lte", {"<=": [{"var": "x"}, 1]}),
    ],
)
def test_fact_predicate_json_logic_comparison_operators(operator: str, expected: dict) -> None:
    pred = FactPredicate(fact="x", operator=operator, value=1)
    assert pred.to_json_logic() == expected


def test_fact_predicate_json_logic_in() -> None:
    pred = FactPredicate(fact="x", operator="in", value=[1, 2, 3])
    assert pred.to_json_logic() == {"in": [{"var": "x"}, [1, 2, 3]]}


def test_fact_predicate_json_logic_not_in() -> None:
    pred = FactPredicate(fact="x", operator="not_in", value=[1, 2])
    assert pred.to_json_logic() == {"!": {"in": [{"var": "x"}, [1, 2]]}}


def test_fact_predicate_json_logic_exists() -> None:
    pred = FactPredicate(fact="x", operator="exists")
    assert pred.to_json_logic() == {"!!": {"var": "x"}}


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


def test_condition_group_json_logic_and() -> None:
    group = ConditionGroup(
        op="and",
        children=[FactPredicate(fact="a", operator="exists"), FactPredicate(fact="b", operator="exists")],
    )
    assert group.to_json_logic() == {"and": [{"!!": {"var": "a"}}, {"!!": {"var": "b"}}]}


def test_condition_group_json_logic_not() -> None:
    group = ConditionGroup(op="not", children=[FactPredicate(fact="a", operator="exists")])
    assert group.to_json_logic() == {"!": {"!!": {"var": "a"}}}


def test_condition_group_json_logic_is_recursive() -> None:
    nested = ConditionGroup(
        op="or",
        children=[
            ConditionGroup(op="and", children=[FactPredicate(fact="a", operator="exists")]),
            FactPredicate(fact="b", operator="eq", value=1),
        ],
    )
    assert nested.to_json_logic() == {
        "or": [{"and": [{"!!": {"var": "a"}}]}, {"==": [{"var": "b"}, 1]}]
    }


def test_condition_round_trips_through_json() -> None:
    """The discriminated union (FactPredicate | ConditionGroup) parses
    back correctly from its own serialized form — exercises the `kind`
    discriminator actually working, not just construction."""
    group = _cond()
    parsed = ConditionGroup.model_validate_json(group.model_dump_json())
    assert parsed.to_json_logic() == group.to_json_logic()


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
