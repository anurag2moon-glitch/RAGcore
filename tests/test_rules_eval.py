"""ragcore.rules.eval: extraction recall/precision/condition-accuracy/
effect-accuracy, verifier agreement rate, hallucination rate, and
conflict-detection recall — Phase 10 §6.
"""

from __future__ import annotations

from ragcore.rules.base import VerificationResult
from ragcore.rules.consistency import RuleConflict
from ragcore.rules.eval import (
    GoldConflictPair,
    GoldRule,
    GoldVerification,
    evaluate_conflict_detection,
    evaluate_extraction,
    evaluate_rules,
    evaluate_verification,
)
from ragcore.rules.types import ConditionGroup, Effect, FactPredicate, Rule


def _rule(rule_id: str, conditions, effect: Effect, *, chunk_ids: tuple[str, ...] = ("c1",)) -> Rule:
    return Rule(
        rule_id=rule_id, domain="test", pack_id="p", pack_version="1", book_id="b1",
        page_start=1, chunk_ids=chunk_ids, verbatim_span="s", language="en", normalized_statement="n",
        conditions=conditions, effect=effect, rule_type="HARD", confidence=0.9,
        extraction_model="m", extraction_version="v1", created_at="now",
    )


_COND_8 = FactPredicate(fact="x", operator="eq", value=8)
_COND_9 = FactPredicate(fact="x", operator="eq", value=9)
_EFFECT_NEG = Effect(category="health", polarity="negative")
_EFFECT_POS = Effect(category="health", polarity="positive")


# -- evaluate_extraction --------------------------------------------------


def test_exact_match_scores_perfect_recall_precision_and_accuracy() -> None:
    gold = [GoldRule(gold_id="g1", chunk_id="c1", conditions=_COND_8, effect=_EFFECT_NEG)]
    extracted = [_rule("r1", _COND_8, _EFFECT_NEG)]

    recall, precision, cond_acc, eff_acc = evaluate_extraction(gold, extracted)

    assert (recall, precision, cond_acc, eff_acc) == (1.0, 1.0, 1.0, 1.0)


def test_nothing_extracted_for_a_gold_chunk_scores_zero_recall_and_none_accuracy() -> None:
    gold = [GoldRule(gold_id="g1", chunk_id="c1", conditions=_COND_8, effect=_EFFECT_NEG)]

    recall, _precision, cond_acc, eff_acc = evaluate_extraction(gold, [])

    assert recall == 0.0
    assert cond_acc is None
    assert eff_acc is None


def test_wrong_effect_but_right_conditions_scores_condition_accuracy_without_recall() -> None:
    gold = [GoldRule(gold_id="g1", chunk_id="c1", conditions=_COND_8, effect=_EFFECT_NEG)]
    extracted = [_rule("r1", _COND_8, _EFFECT_POS)]  # conditions right, effect wrong

    recall, _precision, cond_acc, eff_acc = evaluate_extraction(gold, extracted)

    assert recall == 0.0  # not a full match
    assert cond_acc == 1.0
    assert eff_acc == 0.0


def test_an_extra_extracted_rule_with_no_gold_match_lowers_precision_only() -> None:
    gold = [GoldRule(gold_id="g1", chunk_id="c1", conditions=_COND_8, effect=_EFFECT_NEG)]
    extracted = [
        _rule("r1", _COND_8, _EFFECT_NEG),
        _rule("r2", _COND_9, _EFFECT_POS, chunk_ids=("c2",)),  # no gold rule for c2 at all
    ]

    recall, precision, _cond_acc, _eff_acc = evaluate_extraction(gold, extracted)

    assert recall == 1.0
    assert precision == 0.5  # 1 true positive / 2 extracted


def test_candidate_matching_picks_the_best_rule_not_just_the_first_sharing_a_chunk() -> None:
    """Two gold rules and two extracted rules all share one chunk - the
    matcher must pair each gold rule with its OWN correct counterpart,
    not whichever extracted rule happens to come first in the list."""
    gold = [
        GoldRule(gold_id="g1", chunk_id="c1", conditions=_COND_8, effect=_EFFECT_NEG),
        GoldRule(gold_id="g2", chunk_id="c1", conditions=_COND_9, effect=_EFFECT_POS),
    ]
    extracted = [
        _rule("r_for_g2", _COND_9, _EFFECT_POS),  # listed first, but matches g2
        _rule("r_for_g1", _COND_8, _EFFECT_NEG),  # listed second, but matches g1
    ]

    recall, _precision, cond_acc, eff_acc = evaluate_extraction(gold, extracted)

    assert recall == 1.0
    assert cond_acc == 1.0
    assert eff_acc == 1.0


def test_candidate_matching_prefers_higher_fact_overlap_when_no_exact_match_exists() -> None:
    gold_conditions = ConditionGroup(
        op="and", children=[_COND_8, FactPredicate(fact="y", operator="exists")]
    )
    gold = [GoldRule(gold_id="g1", chunk_id="c1", conditions=gold_conditions, effect=_EFFECT_NEG)]
    closer = _rule("closer", gold_conditions, _EFFECT_POS)  # same conditions, wrong effect
    farther = _rule("farther", _COND_9, _EFFECT_NEG, chunk_ids=("c1",))  # unrelated conditions

    _recall, _precision, cond_acc, eff_acc = evaluate_extraction(gold, [farther, closer])

    # the matcher should have scored against `closer` (full fact overlap),
    # not `farther` (zero overlap) - proven by condition_accuracy being 1.0
    assert cond_acc == 1.0
    assert eff_acc == 0.0


def test_empty_gold_set_returns_zero_recall_precision_and_none_accuracy() -> None:
    recall, precision, cond_acc, eff_acc = evaluate_extraction([], [_rule("r1", _COND_8, _EFFECT_NEG)])
    assert (recall, precision, cond_acc, eff_acc) == (0.0, 0.0, None, None)


# -- evaluate_verification ------------------------------------------------


def test_verifier_agreement_and_hallucination_rate() -> None:
    gold = [
        GoldVerification(rule_id="r1", human_supported=True),
        GoldVerification(rule_id="r2", human_supported=False),
    ]
    verifications = [
        VerificationResult(rule_id="r1", supported=True, reason="ok"),  # agrees
        VerificationResult(rule_id="r2", supported=True, reason="ok"),  # disagrees - verifier missed a hallucination
    ]

    agreement, hallucination = evaluate_verification(gold, verifications)

    assert agreement == 0.5
    assert hallucination == 0.5  # 1 of 2 gold-judged rules is a hallucination


def test_verification_with_no_gold_labels_returns_none() -> None:
    assert evaluate_verification([], []) == (None, None)


def test_verification_gold_entry_with_no_matching_verification_is_skipped() -> None:
    gold = [GoldVerification(rule_id="unverified_rule", human_supported=True)]
    agreement, hallucination = evaluate_verification(gold, [])
    assert (agreement, hallucination) == (None, None)


# -- evaluate_conflict_detection -------------------------------------------


def test_conflict_detection_recall_finds_a_matching_stored_conflict() -> None:
    gold = [GoldConflictPair(gold_id_a="g1", gold_id_b="g2", conflict_type="contradiction")]
    detected = [
        RuleConflict(conflict_id="x", conflict_type="contradiction", rule_ids=("r1", "r2"), rationale="r", winning_rule_id=None)
    ]

    recall = evaluate_conflict_detection(gold, detected, rule_id_by_gold_id={"g1": "r1", "g2": "r2"})

    assert recall == 1.0


def test_conflict_detection_recall_is_order_independent() -> None:
    gold = [GoldConflictPair(gold_id_a="g1", gold_id_b="g2", conflict_type="contradiction")]
    detected = [
        RuleConflict(conflict_id="x", conflict_type="contradiction", rule_ids=("r2", "r1"), rationale="r", winning_rule_id=None)
    ]

    recall = evaluate_conflict_detection(gold, detected, rule_id_by_gold_id={"g1": "r1", "g2": "r2"})

    assert recall == 1.0


def test_conflict_detection_recall_misses_a_wrong_conflict_type() -> None:
    gold = [GoldConflictPair(gold_id_a="g1", gold_id_b="g2", conflict_type="exception")]
    detected = [
        RuleConflict(conflict_id="x", conflict_type="contradiction", rule_ids=("r1", "r2"), rationale="r", winning_rule_id=None)
    ]

    recall = evaluate_conflict_detection(gold, detected, rule_id_by_gold_id={"g1": "r1", "g2": "r2"})

    assert recall == 0.0


def test_conflict_detection_recall_with_no_gold_pairs_returns_none() -> None:
    assert evaluate_conflict_detection([], [], rule_id_by_gold_id={}) is None


def test_conflict_detection_recall_handles_an_ungraded_gold_pair() -> None:
    """A gold pair whose gold_id never made it into rule_id_by_gold_id
    (e.g. the expected rule was never actually extracted at all) must
    count as a miss, not raise."""
    gold = [GoldConflictPair(gold_id_a="g1", gold_id_b="never_extracted", conflict_type="contradiction")]
    recall = evaluate_conflict_detection(gold, [], rule_id_by_gold_id={"g1": "r1"})
    assert recall == 0.0


# -- evaluate_rules (the combined report) ---------------------------------


def test_evaluate_rules_combines_every_metric_into_one_report() -> None:
    gold_rules = [GoldRule(gold_id="g1", chunk_id="c1", conditions=_COND_8, effect=_EFFECT_NEG)]
    extracted = [_rule("r1", _COND_8, _EFFECT_NEG)]
    gold_verifications = [GoldVerification(rule_id="r1", human_supported=True)]
    verifications = [VerificationResult(rule_id="r1", supported=True, reason="ok")]

    report = evaluate_rules(
        gold_rules=gold_rules, extracted=extracted,
        gold_verifications=gold_verifications, verifications=verifications,
    )

    assert report.gold_count == 1
    assert report.extracted_count == 1
    assert report.recall == 1.0
    assert report.verifier_agreement_rate == 1.0
    assert report.conflict_detection_recall is None  # no gold conflicts supplied
