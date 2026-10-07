"""match_with_provenance: condition matching, the verified-only-by-
default filter (Phase 10 §4), and grouping matched rules by their
already-stored conflicts into an "effective" (precedence-resolved)
subset — Phase 10 §5.
"""

from __future__ import annotations

from ragcore.rules.consistency import detect_conflicts
from ragcore.rules.serving import match_with_provenance
from ragcore.rules.sqlite_store import SqliteRuleStore
from ragcore.rules.types import ConditionGroup, Effect, FactPredicate, Rule


def _rule(
    rule_id: str, conditions, effect: Effect, *, rule_type: str = "HARD",
    book_id: str = "b1", review_status: str = "verified",
) -> Rule:
    return Rule(
        rule_id=rule_id, domain="test", pack_id="p", pack_version="1", book_id=book_id,
        page_start=1, chunk_ids=("c:0",), verbatim_span=f"span {rule_id}", language="en",
        normalized_statement="n", conditions=conditions, effect=effect, rule_type=rule_type,
        confidence=0.9, extraction_model="m", extraction_version="v1", created_at="now",
        review_status=review_status,
    )


_COND_8 = FactPredicate(fact="planet.mars.house", operator="eq", value=8)
_EFFECT_NEG = Effect(category="health", polarity="negative")
_EFFECT_POS = Effect(category="health", polarity="positive")
_FACTS = {"planet.mars.house": 8, "planet.mars.retrograde": True}


def test_only_conditions_actually_holding_are_matched() -> None:
    store = SqliteRuleStore(":memory:")
    holds = _rule("holds", _COND_8, _EFFECT_NEG)
    doesnt = _rule("doesnt", FactPredicate(fact="planet.saturn.house", operator="eq", value=1), _EFFECT_POS)

    result = match_with_provenance([holds, doesnt], _FACTS, store=store)

    assert [m.rule.rule_id for m in result.matched] == ["holds"]
    assert result.effective == (holds,)


def test_unverified_rules_are_excluded_by_default() -> None:
    store = SqliteRuleStore(":memory:")
    verified = _rule("verified", _COND_8, _EFFECT_NEG)
    auto = _rule("auto", _COND_8, _EFFECT_POS, review_status="auto", book_id="b2")

    result = match_with_provenance([verified, auto], _FACTS, store=store)

    assert [m.rule.rule_id for m in result.matched] == ["verified"]


def test_include_unverified_opts_into_everything_that_matches() -> None:
    store = SqliteRuleStore(":memory:")
    verified = _rule("verified", _COND_8, _EFFECT_NEG)
    auto = _rule("auto", _COND_8, _EFFECT_POS, review_status="auto", book_id="b2")

    result = match_with_provenance([verified, auto], _FACTS, store=store, include_unverified=True)

    assert {m.rule.rule_id for m in result.matched} == {"verified", "auto"}


def test_an_exception_rule_overrides_the_general_rule_it_conflicts_with() -> None:
    general = _rule("general", _COND_8, _EFFECT_NEG)
    specific_cond = ConditionGroup(
        op="and", children=[_COND_8, FactPredicate(fact="planet.mars.retrograde", operator="exists")]
    )
    exception = _rule("exception", specific_cond, _EFFECT_POS, rule_type="EXCEPTION", book_id="b2")

    store = SqliteRuleStore(":memory:")
    store.put_conflicts(detect_conflicts([general, exception]))

    result = match_with_provenance([general, exception], _FACTS, store=store)

    by_id = {m.rule.rule_id: m for m in result.matched}
    assert by_id["general"].is_overridden is True
    assert by_id["exception"].is_overridden is False
    assert by_id["exception"].overrides == ("general",)
    assert result.effective == (exception,)


def test_without_running_conflict_detection_everything_matched_is_effective() -> None:
    """A rule base that's never had detect_conflicts()/put_conflicts()
    run over it still serves correct, conflict-unaware results - a
    safe degradation, not a crash or a silently wrong answer."""
    general = _rule("general", _COND_8, _EFFECT_NEG)
    specific_cond = ConditionGroup(
        op="and", children=[_COND_8, FactPredicate(fact="planet.mars.retrograde", operator="exists")]
    )
    exception = _rule("exception", specific_cond, _EFFECT_POS, rule_type="EXCEPTION", book_id="b2")

    store = SqliteRuleStore(":memory:")  # conflicts never detected/stored

    result = match_with_provenance([general, exception], _FACTS, store=store)

    assert {r.rule_id for r in result.effective} == {"general", "exception"}


def test_a_conflict_with_a_rule_that_didnt_match_is_not_surfaced() -> None:
    """A stored conflict naming a rule that didn't match THIS fact set
    is irrelevant to this query - it must not appear in the matched
    rule's own conflicts list nor affect is_overridden."""
    general = _rule("general", _COND_8, _EFFECT_NEG)
    specific_cond = ConditionGroup(
        op="and", children=[_COND_8, FactPredicate(fact="planet.mars.retrograde", operator="exists")]
    )
    exception = _rule("exception", specific_cond, _EFFECT_POS, rule_type="EXCEPTION", book_id="b2")

    store = SqliteRuleStore(":memory:")
    store.put_conflicts(detect_conflicts([general, exception]))

    facts_without_retrograde = {"planet.mars.house": 8}  # exception's condition doesn't hold
    result = match_with_provenance([general, exception], facts_without_retrograde, store=store)

    assert [m.rule.rule_id for m in result.matched] == ["general"]
    assert result.matched[0].conflicts == ()
    assert result.matched[0].is_overridden is False
    assert result.effective == (general,)


def test_no_rules_match_produces_an_empty_result() -> None:
    store = SqliteRuleStore(":memory:")
    rule = _rule("r1", FactPredicate(fact="planet.saturn.house", operator="eq", value=1), _EFFECT_POS)

    result = match_with_provenance([rule], _FACTS, store=store)

    assert result.matched == ()
    assert result.effective == ()
