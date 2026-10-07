"""ragcore.rules.evaluator: evaluate_condition's missing-fact semantics
(the authoritative behavior test_rules_types.py's JSON-Logic
cross-check is verified against), validate_facts, and match_rules —
Phase 10 §5.
"""

from __future__ import annotations

from ragcore.rules.domain import FactTypeSpec
from ragcore.rules.evaluator import evaluate_condition, match_rules, validate_facts
from ragcore.rules.types import ConditionGroup, Effect, FactPredicate, Rule


def _rule(rule_id: str, conditions, *, confidence: float = 0.9) -> Rule:
    return Rule(
        rule_id=rule_id,
        domain="test",
        pack_id="p",
        pack_version="1",
        book_id="b",
        page_start=1,
        chunk_ids=("b:0",),
        verbatim_span="text",
        language="en",
        normalized_statement="n",
        conditions=conditions,
        effect=Effect(category="health", polarity="negative"),
        rule_type="HARD",
        confidence=confidence,
        extraction_model="m",
        extraction_version="v1",
        created_at="now",
    )


# -- evaluate_condition: leaf predicates -----------------------------------


def test_eq_matches_the_exact_value() -> None:
    pred = FactPredicate(fact="x", operator="eq", value=8)
    assert evaluate_condition(pred, {"x": 8}) is True
    assert evaluate_condition(pred, {"x": 5}) is False


def test_ne_matches_a_different_value() -> None:
    pred = FactPredicate(fact="x", operator="ne", value=8)
    assert evaluate_condition(pred, {"x": 5}) is True
    assert evaluate_condition(pred, {"x": 8}) is False


def test_in_matches_membership() -> None:
    pred = FactPredicate(fact="x", operator="in", value=[1, 2, 3])
    assert evaluate_condition(pred, {"x": 2}) is True
    assert evaluate_condition(pred, {"x": 9}) is False


def test_not_in_matches_non_membership() -> None:
    pred = FactPredicate(fact="x", operator="not_in", value=[1, 2, 3])
    assert evaluate_condition(pred, {"x": 9}) is True
    assert evaluate_condition(pred, {"x": 2}) is False


def test_gt_gte_lt_lte() -> None:
    assert evaluate_condition(FactPredicate(fact="x", operator="gt", value=5), {"x": 6}) is True
    assert evaluate_condition(FactPredicate(fact="x", operator="gt", value=5), {"x": 5}) is False
    assert evaluate_condition(FactPredicate(fact="x", operator="gte", value=5), {"x": 5}) is True
    assert evaluate_condition(FactPredicate(fact="x", operator="lt", value=5), {"x": 4}) is True
    assert evaluate_condition(FactPredicate(fact="x", operator="lte", value=5), {"x": 5}) is True


def test_exists_checks_presence_not_truthiness() -> None:
    pred = FactPredicate(fact="x", operator="exists")
    assert evaluate_condition(pred, {"x": 0}) is True
    assert evaluate_condition(pred, {"x": False}) is True
    assert evaluate_condition(pred, {"x": ""}) is True
    assert evaluate_condition(pred, {}) is False


# -- evaluate_condition: missing-fact closed-world semantics -----------


def test_every_non_exists_operator_is_false_when_fact_is_missing() -> None:
    """The deliberate closed-world choice this module's own docstring
    states: a missing fact is False for every operator except exists,
    including ne/not_in which might look "true" under an open-world
    reading of absence."""
    for operator, value in [
        ("eq", 1), ("ne", 1), ("in", [1, 2]), ("not_in", [1, 2]),
        ("gt", 1), ("gte", 1), ("lt", 1), ("lte", 1),
    ]:
        pred = FactPredicate(fact="missing", operator=operator, value=value)
        assert evaluate_condition(pred, {}) is False, f"operator {operator!r} should be False when missing"


# -- evaluate_condition: groups ---------------------------------------


def test_and_requires_every_child_true() -> None:
    group = ConditionGroup(
        op="and",
        children=[FactPredicate(fact="a", operator="exists"), FactPredicate(fact="b", operator="exists")],
    )
    assert evaluate_condition(group, {"a": 1, "b": 1}) is True
    assert evaluate_condition(group, {"a": 1}) is False


def test_or_requires_one_child_true() -> None:
    group = ConditionGroup(
        op="or",
        children=[FactPredicate(fact="a", operator="exists"), FactPredicate(fact="b", operator="exists")],
    )
    assert evaluate_condition(group, {"a": 1}) is True
    assert evaluate_condition(group, {}) is False


def test_not_inverts_its_single_child() -> None:
    group = ConditionGroup(op="not", children=[FactPredicate(fact="a", operator="eq", value=1)])
    assert evaluate_condition(group, {"a": 1}) is False
    assert evaluate_condition(group, {"a": 2}) is True


def test_nested_groups_evaluate_recursively() -> None:
    nested = ConditionGroup(
        op="and",
        children=[
            ConditionGroup(
                op="or",
                children=[
                    FactPredicate(fact="a", operator="eq", value=1),
                    FactPredicate(fact="b", operator="eq", value=2),
                ],
            ),
            FactPredicate(fact="c", operator="exists"),
        ],
    )
    assert evaluate_condition(nested, {"a": 1, "c": 1}) is True
    assert evaluate_condition(nested, {"b": 2, "c": 1}) is True
    assert evaluate_condition(nested, {"a": 1}) is False  # c missing


# -- validate_facts -------------------------------------------------------


def test_validate_facts_returns_no_errors_for_well_typed_known_facts() -> None:
    specs = (FactTypeSpec(fact_key="x", value_type="int"),)
    assert validate_facts({"x": 8}, specs) == []


def test_validate_facts_does_not_require_every_known_fact_to_be_present() -> None:
    specs = (
        FactTypeSpec(fact_key="x", value_type="int"),
        FactTypeSpec(fact_key="y", value_type="int"),
    )
    assert validate_facts({"x": 8}, specs) == []


def test_validate_facts_flags_an_unknown_fact() -> None:
    errors = validate_facts({"unknown": 1}, ())
    assert len(errors) == 1
    assert "unknown" in errors[0]


def test_validate_facts_flags_a_mistyped_fact() -> None:
    specs = (FactTypeSpec(fact_key="x", value_type="int"),)
    errors = validate_facts({"x": "not an int"}, specs)
    assert len(errors) == 1
    assert "x" in errors[0]


def test_validate_facts_reports_one_error_per_bad_fact() -> None:
    specs = (FactTypeSpec(fact_key="x", value_type="int"),)
    errors = validate_facts({"x": "bad", "unknown": 1}, specs)
    assert len(errors) == 2


# -- match_rules --------------------------------------------------------


def test_match_rules_returns_only_rules_whose_conditions_hold() -> None:
    rule_a = _rule("rule:a", FactPredicate(fact="x", operator="eq", value=1))
    rule_b = _rule("rule:b", FactPredicate(fact="x", operator="eq", value=2))

    matched = match_rules([rule_a, rule_b], {"x": 1})

    assert [r.rule_id for r in matched] == ["rule:a"]


def test_match_rules_returns_empty_list_when_nothing_matches() -> None:
    rule_a = _rule("rule:a", FactPredicate(fact="x", operator="eq", value=1))
    assert match_rules([rule_a], {"x": 99}) == []


def test_match_rules_preserves_input_order() -> None:
    always_true = FactPredicate(fact="x", operator="exists")
    rule_a = _rule("rule:a", always_true)
    rule_b = _rule("rule:b", always_true)
    rule_c = _rule("rule:c", always_true)

    matched = match_rules([rule_c, rule_a, rule_b], {"x": 1})

    assert [r.rule_id for r in matched] == ["rule:c", "rule:a", "rule:b"]
