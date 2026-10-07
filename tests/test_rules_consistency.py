"""find_duplicates / find_contradictions / find_exceptions /
detect_conflicts / resolve_precedence — Phase 10 §3. Covers the
flat-conjunction limitation (or/not conditions are skipped by
contradiction/exception detection, not silently mishandled) and the
exception-over-rule vs. specific-over-general precedence rules.
"""

from __future__ import annotations

from ragcore.rules.consistency import (
    detect_conflicts,
    find_contradictions,
    find_duplicates,
    find_exceptions,
    resolve_precedence,
)
from ragcore.rules.types import ConditionGroup, Effect, FactPredicate, Rule


def _rule(
    rule_id: str, conditions, effect: Effect, *, rule_type: str = "HARD",
    domain: str = "test", book_id: str = "b1",
) -> Rule:
    return Rule(
        rule_id=rule_id, domain=domain, pack_id="p", pack_version="1", book_id=book_id,
        page_start=1, chunk_ids=("c:0",), verbatim_span=f"span {rule_id}", language="en",
        normalized_statement="n", conditions=conditions, effect=effect, rule_type=rule_type,
        confidence=0.9, extraction_model="m", extraction_version="v1", created_at="now",
    )


_COND_8 = FactPredicate(fact="planet.mars.house", operator="eq", value=8)
_EFFECT_NEG = Effect(category="health", polarity="negative")
_EFFECT_POS = Effect(category="health", polarity="positive")


# -- find_duplicates -------------------------------------------------------


def test_identical_conditions_and_effect_across_books_is_a_duplicate() -> None:
    r1 = _rule("r1", _COND_8, _EFFECT_NEG, book_id="book1")
    r2 = _rule("r2", FactPredicate(fact="planet.mars.house", operator="eq", value=8), _EFFECT_NEG, book_id="book2")

    dups = find_duplicates([r1, r2])

    assert len(dups) == 1
    assert dups[0].rule_ids == ("r1", "r2")
    assert dups[0].conflict_type == "duplicate"
    assert dups[0].winning_rule_id is None


def test_different_effects_are_not_duplicates() -> None:
    r1 = _rule("r1", _COND_8, _EFFECT_NEG)
    r2 = _rule("r2", _COND_8, _EFFECT_POS)
    assert find_duplicates([r1, r2]) == []


def test_different_domains_are_never_compared() -> None:
    r1 = _rule("r1", _COND_8, _EFFECT_NEG, domain="domain_a")
    r2 = _rule("r2", _COND_8, _EFFECT_NEG, domain="domain_b")
    assert find_duplicates([r1, r2]) == []


def test_a_group_of_three_duplicates_produces_one_conflict_per_pair() -> None:
    rules = [_rule(f"r{i}", _COND_8, _EFFECT_NEG, book_id=f"b{i}") for i in range(3)]
    dups = find_duplicates(rules)
    assert len(dups) == 3  # C(3,2)
    assert {d.rule_ids for d in dups} == {("r0", "r1"), ("r0", "r2"), ("r1", "r2")}


def test_duplicate_detection_is_order_independent_in_conflict_id() -> None:
    r1 = _rule("r1", _COND_8, _EFFECT_NEG)
    r2 = _rule("r2", _COND_8, _EFFECT_NEG)
    forward = find_duplicates([r1, r2])[0].conflict_id
    backward = find_duplicates([r2, r1])[0].conflict_id
    assert forward == backward


# -- find_contradictions ---------------------------------------------------


def test_same_conditions_opposite_polarity_same_category_is_a_contradiction() -> None:
    r1 = _rule("r1", _COND_8, _EFFECT_NEG)
    r2 = _rule("r2", _COND_8, _EFFECT_POS)

    contras = find_contradictions([r1, r2])

    assert len(contras) == 1
    assert contras[0].rule_ids == ("r1", "r2")
    assert contras[0].conflict_type == "contradiction"


def test_same_conditions_same_polarity_is_not_a_contradiction() -> None:
    r1 = _rule("r1", _COND_8, _EFFECT_NEG)
    r2 = _rule("r2", _COND_8, Effect(category="health", polarity="negative"))
    assert find_contradictions([r1, r2]) == []


def test_same_conditions_different_category_is_not_a_contradiction() -> None:
    r1 = _rule("r1", _COND_8, _EFFECT_NEG)
    r2 = _rule("r2", _COND_8, Effect(category="wealth", polarity="positive"))
    assert find_contradictions([r1, r2]) == []


def test_a_contradiction_with_equal_specificity_is_unresolved() -> None:
    r1 = _rule("r1", _COND_8, _EFFECT_NEG)
    r2 = _rule("r2", _COND_8, _EFFECT_POS)
    contra = find_contradictions([r1, r2])[0]
    assert contra.winning_rule_id is None


def test_an_or_condition_is_not_comparable_for_contradiction() -> None:
    or_cond = ConditionGroup(op="or", children=[_COND_8, FactPredicate(fact="y", operator="exists")])
    r1 = _rule("r1", _COND_8, _EFFECT_NEG)
    r2 = _rule("r2", or_cond, _EFFECT_POS)
    assert find_contradictions([r1, r2]) == []


# -- find_exceptions --------------------------------------------------------


def test_an_exception_rule_with_a_superset_of_conditions_carves_out_a_case() -> None:
    general = _rule("general", _COND_8, _EFFECT_NEG)
    specific_cond = ConditionGroup(
        op="and", children=[_COND_8, FactPredicate(fact="planet.mars.retrograde", operator="exists")]
    )
    exception = _rule("exception", specific_cond, _EFFECT_POS, rule_type="EXCEPTION")

    excs = find_exceptions([general, exception])

    assert len(excs) == 1
    assert excs[0].rule_ids == ("general", "exception")
    assert excs[0].winning_rule_id == "exception"
    assert excs[0].conflict_type == "exception"


def test_a_cancellation_rule_also_counts_as_an_exception_relationship() -> None:
    general = _rule("general", _COND_8, _EFFECT_NEG)
    specific_cond = ConditionGroup(
        op="and", children=[_COND_8, FactPredicate(fact="planet.saturn.house", operator="eq", value=1)]
    )
    cancellation = _rule("cancellation", specific_cond, _EFFECT_POS, rule_type="CANCELLATION")

    excs = find_exceptions([general, cancellation])

    assert excs[0].winning_rule_id == "cancellation"


def test_a_hard_rule_with_a_superset_of_conditions_is_not_an_exception() -> None:
    general = _rule("general", _COND_8, _EFFECT_NEG)
    specific_cond = ConditionGroup(
        op="and", children=[_COND_8, FactPredicate(fact="planet.mars.retrograde", operator="exists")]
    )
    not_an_exception = _rule("specific_hard", specific_cond, _EFFECT_POS, rule_type="HARD")

    assert find_exceptions([general, not_an_exception]) == []


def test_equal_condition_sets_are_not_an_exception_relationship() -> None:
    """A proper SUPERSET is required - identical conditions means
    there's nothing more specific being carved out, so this is a
    contradiction (or duplicate), never an exception."""
    r1 = _rule("r1", _COND_8, _EFFECT_NEG)
    r2 = _rule("r2", _COND_8, _EFFECT_POS, rule_type="EXCEPTION")
    assert find_exceptions([r1, r2]) == []


def test_an_or_condition_is_not_comparable_for_exceptions() -> None:
    general = _rule("general", _COND_8, _EFFECT_NEG)
    or_cond = ConditionGroup(op="or", children=[_COND_8, FactPredicate(fact="y", operator="exists")])
    exception = _rule("exception", or_cond, _EFFECT_POS, rule_type="EXCEPTION")
    assert find_exceptions([general, exception]) == []


# -- detect_conflicts / resolve_precedence ----------------------------------


def test_detect_conflicts_combines_all_three_kinds() -> None:
    r1 = _rule("r1", _COND_8, _EFFECT_NEG, book_id="b1")
    r2 = _rule("r2", _COND_8, _EFFECT_NEG, book_id="b2")  # duplicate of r1
    r3 = _rule("r3", _COND_8, _EFFECT_POS, book_id="b3")  # contradicts r1/r2
    specific_cond = ConditionGroup(
        op="and", children=[_COND_8, FactPredicate(fact="planet.mars.retrograde", operator="exists")]
    )
    r4 = _rule("r4", specific_cond, _EFFECT_POS, rule_type="EXCEPTION", book_id="b4")  # exception to r1/r2/r3

    conflicts = detect_conflicts([r1, r2, r3, r4])

    types = {c.conflict_type for c in conflicts}
    assert types == {"duplicate", "contradiction", "exception"}


def test_resolve_precedence_for_an_exception_returns_its_stored_winner() -> None:
    general = _rule("general", _COND_8, _EFFECT_NEG)
    specific_cond = ConditionGroup(
        op="and", children=[_COND_8, FactPredicate(fact="planet.mars.retrograde", operator="exists")]
    )
    exception = _rule("exception", specific_cond, _EFFECT_POS, rule_type="EXCEPTION")
    conflict = find_exceptions([general, exception])[0]

    rules_by_id = {"general": general, "exception": exception}
    assert resolve_precedence(conflict, rules_by_id) == "exception"


def test_resolve_precedence_for_a_duplicate_is_always_none() -> None:
    r1 = _rule("r1", _COND_8, _EFFECT_NEG, book_id="b1")
    r2 = _rule("r2", _COND_8, _EFFECT_NEG, book_id="b2")
    conflict = find_duplicates([r1, r2])[0]
    assert resolve_precedence(conflict, {"r1": r1, "r2": r2}) is None


def test_resolve_precedence_for_contradiction_breaks_a_specificity_tie_when_present() -> None:
    """find_contradictions itself only ever groups rules with IDENTICAL
    flat condition sets (same specificity, by construction), so it
    never emits a resolvable contradiction - but resolve_precedence's
    own specific-over-general logic is tested directly here, in
    isolation from that caller, with a hand-built conflict record
    whose two rules genuinely differ in specificity."""
    from ragcore.rules.consistency import RuleConflict

    general = _rule("general", _COND_8, _EFFECT_NEG)
    specific_cond = ConditionGroup(
        op="and", children=[_COND_8, FactPredicate(fact="planet.mars.retrograde", operator="exists")]
    )
    specific = _rule("specific", specific_cond, _EFFECT_POS)
    conflict = RuleConflict(
        conflict_id="ruleconflict:test", conflict_type="contradiction",
        rule_ids=("general", "specific"), rationale="test", winning_rule_id=None,
    )

    winner = resolve_precedence(conflict, {"general": general, "specific": specific})

    assert winner == "specific"
