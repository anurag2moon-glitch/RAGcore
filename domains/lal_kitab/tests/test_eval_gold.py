"""Phase 10 §6 CI regression gate: ragcore.rules.eval run against the
hand-labeled 3-chapter Lal Kitab gold set (tests/gold.py). Not a live-
quality benchmark against a real Claude extraction run — everything
here is deterministic (the "simulated extraction" in gold.py is a
fixed fixture, not an LLM call), so this instead guards against a
regression in the LalKitabOntologyNormalizer-driven rule construction
path or in ragcore.rules.eval's own scoring logic ever silently
breaking.

The fixture is deliberately not a trivial 100%-on-everything scenario
— `gold_ch3_career` is never extracted (a missed rule) and one
hallucinated rule is included with no matching gold rule, specifically
so this gate can tell "found everything correctly" apart from "found
nothing," matching how `tests/integration/test_eval_quality.py` in the
main ragcore repo already compares against a committed baseline rather
than asserting everything is perfect by construction.
"""

from __future__ import annotations

from gold import build_gold_fixture
from lal_kitab import LalKitabPack

from ragcore.rules.consistency import detect_conflicts
from ragcore.rules.eval import evaluate_rules


def test_rule_eval_against_the_hand_labeled_gold_set() -> None:
    pack = LalKitabPack()
    gold_rules, extracted, gold_verifications, verifications, gold_conflicts, rule_id_by_gold_id = (
        build_gold_fixture(pack)
    )
    detected_conflicts = detect_conflicts(extracted)

    report = evaluate_rules(
        gold_rules=gold_rules, extracted=extracted,
        gold_verifications=gold_verifications, verifications=verifications,
        gold_conflicts=gold_conflicts, detected_conflicts=detected_conflicts,
        rule_id_by_gold_id=rule_id_by_gold_id,
    )

    # 3 of 4 gold rules are found exactly right (gold_ch3_career is the
    # deliberate miss) - any further drop is a real regression.
    assert report.recall == 0.75
    # 3 of 4 extracted rules are true positives (the hallucinated rule
    # is the one false positive).
    assert report.precision == 0.75
    # Every rule that WAS extracted has exactly correct conditions/effect
    # in this fixture - a drop here means something broke the ontology-
    # to-schema construction path, not just a "missed" rule.
    assert report.condition_accuracy == 1.0
    assert report.effect_accuracy == 1.0
    # The simulated verifier agrees with every human label by construction.
    assert report.verifier_agreement_rate == 1.0
    # 1 of 4 human-judged rules is a hallucination (the one the verifier
    # correctly flagged as unsupported).
    assert report.hallucination_rate == 0.25
    # The general-vs-exception conflict between ch1 and ch2 must be
    # detected by ragcore.rules.consistency.detect_conflicts.
    assert report.conflict_detection_recall == 1.0


def test_the_exception_conflict_in_the_gold_set_resolves_to_the_exception_rule() -> None:
    """A finer-grained check than conflict_detection_recall alone: not
    just THAT the conflict is detected, but that precedence resolves
    to the EXCEPTION rule, not the general one it carves from."""
    pack = LalKitabPack()
    _gold_rules, extracted, *_rest = build_gold_fixture(pack)
    conflicts = detect_conflicts(extracted)

    exception_conflicts = [c for c in conflicts if c.conflict_type == "exception"]
    assert len(exception_conflicts) == 1

    ch2_rule = next(r for r in extracted if r.rule_type == "EXCEPTION")
    assert exception_conflicts[0].winning_rule_id == ch2_rule.rule_id
