"""Evaluation harness (Phase 10 §6): "Hand-labelled gold set for 3
chapters: measure rule recall, precision, condition-accuracy and
effect-accuracy. Verifier agreement rate, hallucination rate (rules
with unsupported spans), conflict-detection recall. Regression gate in
CI on the gold set."

Domain-agnostic by design, like everything else in `ragcore.rules` —
`GoldRule` is just a `chunk_id` + structured `conditions`/`effect`, no
domain vocabulary anywhere in this module. The actual hand-labeled
gold set for Lal Kitab lives in `domains/lal_kitab/` (a sibling
package, per ADR 0013), with its own CI job running this module's
functions against it — the same split `ragcore.eval` already has
between "general-purpose harness" and "a consumer's own gold set."

Matching granularity, stated precisely: a gold rule counts as **fully
recalled** only if some extracted rule on the same `chunk_id` has
*structurally identical* `conditions` and `effect` (the same equality
`ragcore.rules.consistency.find_duplicates` uses). `condition_accuracy`/
`effect_accuracy` are finer-grained than recall: among gold rules where
*something* was extracted for the same chunk (a "candidate match", even
if wrong), what fraction have correct conditions, and what fraction
have a correct effect — independently, since a rule can get one right
and the other wrong. A gold rule with no candidate match at all
(nothing extracted for that chunk) contributes to recall's denominator
but not to condition/effect accuracy's — there's nothing to score the
accuracy of.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from ragcore.rules.base import VerificationResult
    from ragcore.rules.consistency import ConflictType, RuleConflict
    from ragcore.rules.types import Condition, Effect, Rule


@dataclass(frozen=True)
class GoldRule:
    gold_id: str
    chunk_id: str
    conditions: Condition
    effect: Effect


@dataclass(frozen=True)
class GoldVerification:
    """A human reviewer's own supported/unsupported judgment for one
    already-extracted rule — compared against that rule's automated
    VerificationResult to measure verifier agreement, and used
    independently to measure hallucination rate (the fraction of
    gold-judged rules a human says the span does NOT actually
    support)."""

    rule_id: str
    human_supported: bool


@dataclass(frozen=True)
class GoldConflictPair:
    gold_id_a: str
    gold_id_b: str
    conflict_type: ConflictType


@dataclass(frozen=True)
class RuleEvalReport:
    gold_count: int
    extracted_count: int
    recall: float
    precision: float
    condition_accuracy: float | None
    effect_accuracy: float | None
    verifier_agreement_rate: float | None
    hallucination_rate: float | None
    conflict_detection_recall: float | None


def _structurally_equal(a: Condition | Effect, b: Condition | Effect) -> bool:
    return a.model_dump() == b.model_dump()


def _collect_leaf_facts(condition: Condition) -> frozenset[tuple[str, str, object]]:
    """Every (fact, operator, value) leaf anywhere in the tree,
    regardless of AND/OR/NOT structure — unlike
    ragcore.rules.consistency._flat_facts (which demands a flat
    conjunction and returns None otherwise), this is only ever used
    for *similarity ranking* between candidate rules sharing a chunk,
    never for a correctness judgment, so being lenient about structure
    here is deliberate, not a shortcut that could mask a real bug."""
    from ragcore.rules.types import ConditionGroup, FactPredicate

    if isinstance(condition, FactPredicate):
        value = tuple(condition.value) if isinstance(condition.value, list) else condition.value
        return frozenset({(condition.fact, condition.operator, value)})
    if isinstance(condition, ConditionGroup):
        facts: set[tuple[str, str, object]] = set()
        for child in condition.children:
            facts |= _collect_leaf_facts(child)
        return frozenset(facts)
    return frozenset()  # pragma: no cover - exhaustive over the Condition union


def _fully_matches(gold: GoldRule, rule: Rule) -> bool:
    return _structurally_equal(gold.conditions, rule.conditions) and _structurally_equal(gold.effect, rule.effect)


def _candidate_match(gold: GoldRule, extracted: Sequence[Rule]) -> Rule | None:
    """The best-matching extracted rule sharing gold's chunk_id, when
    more than one exists — a chunk commonly yields several distinct
    rules, and picking arbitrarily the first one in list order would
    score condition/effect accuracy against the wrong rule whenever
    gold isn't that first one. Prefers an exact full match outright;
    otherwise prefers the rule with the most overlapping leaf facts
    with gold's own conditions (a similarity heuristic, not a
    correctness judgment — see _collect_leaf_facts).

    Returns None — "nothing extracted for this gold rule" — when even
    the best same-chunk candidate shares **zero** facts with gold's
    own conditions, rather than forcing a pairing onto some unrelated
    rule that merely happens to share a chunk. A multi-rule chunk with
    one gold rule genuinely missed (nothing resembling it extracted)
    must not get scored as "attempted, but wrong" against whichever
    unrelated sibling rule happened to also come from that chunk —
    confirmed by a real false failure this produced before the fix,
    caught by the Lal Kitab gold-set regression test
    (domains/lal_kitab/tests/test_eval_gold.py), not hand-derived."""
    same_chunk = [r for r in extracted if gold.chunk_id in r.chunk_ids]
    if not same_chunk:
        return None

    exact = [r for r in same_chunk if _fully_matches(gold, r)]
    if exact:
        return exact[0]

    gold_facts = _collect_leaf_facts(gold.conditions)
    best = max(same_chunk, key=lambda r: len(gold_facts & _collect_leaf_facts(r.conditions)))
    if not (gold_facts & _collect_leaf_facts(best.conditions)):
        return None
    return best


def evaluate_extraction(
    gold: Sequence[GoldRule], extracted: Sequence[Rule]
) -> tuple[float, float, float | None, float | None]:
    """Returns (recall, precision, condition_accuracy, effect_accuracy)."""
    if not gold:
        return (0.0, 0.0, None, None)

    candidates = [(g, _candidate_match(g, extracted)) for g in gold]
    full_matches = [g for g, c in candidates if c is not None and _fully_matches(g, c)]
    recall = len(full_matches) / len(gold)

    matched_gold_rule_ids = {c.rule_id for g, c in candidates if c is not None and _fully_matches(g, c)}
    precision = (
        sum(1 for rule in extracted if rule.rule_id in matched_gold_rule_ids) / len(extracted)
        if extracted else 0.0
    )

    with_candidate = [(g, c) for g, c in candidates if c is not None]
    if with_candidate:
        condition_accuracy = sum(
            1 for g, c in with_candidate if _structurally_equal(g.conditions, c.conditions)
        ) / len(with_candidate)
        effect_accuracy = sum(
            1 for g, c in with_candidate if _structurally_equal(g.effect, c.effect)
        ) / len(with_candidate)
    else:
        condition_accuracy = None
        effect_accuracy = None

    return (recall, precision, condition_accuracy, effect_accuracy)


def evaluate_verification(
    gold: Sequence[GoldVerification], verifications: Sequence[VerificationResult]
) -> tuple[float | None, float | None]:
    """Returns (verifier_agreement_rate, hallucination_rate). Both None
    if `gold` is empty - there's nothing to measure agreement or
    hallucination against."""
    if not gold:
        return (None, None)

    verifications_by_rule = {v.rule_id: v for v in verifications}
    agreements = 0
    judged = 0
    hallucinated = 0
    for g in gold:
        verification = verifications_by_rule.get(g.rule_id)
        if verification is None:
            continue
        judged += 1
        if verification.supported == g.human_supported:
            agreements += 1
        if not g.human_supported:
            hallucinated += 1

    if judged == 0:
        return (None, None)
    return (agreements / judged, hallucinated / judged)


def evaluate_conflict_detection(
    gold: Sequence[GoldConflictPair],
    detected: Sequence[RuleConflict],
    *,
    rule_id_by_gold_id: dict[str, str],
) -> float | None:
    """conflict_detection_recall: the fraction of gold conflict pairs
    for which a matching RuleConflict (same two rule_ids, same
    conflict_type) was actually detected. `rule_id_by_gold_id` maps
    each GoldConflictPair's gold_id to the real rule_id it ended up
    extracted as (the gold set is authored in terms of gold_ids, which
    are stable across re-extraction; rule_ids are content-addressed
    and only known once extraction has actually run)."""
    if not gold:
        return None

    detected_keys = {
        (conflict.conflict_type, *sorted(conflict.rule_ids)) for conflict in detected
    }
    hits = 0
    for pair in gold:
        rule_a = rule_id_by_gold_id.get(pair.gold_id_a)
        rule_b = rule_id_by_gold_id.get(pair.gold_id_b)
        if rule_a is None or rule_b is None:
            continue
        key = (pair.conflict_type, *sorted((rule_a, rule_b)))
        if key in detected_keys:
            hits += 1
    return hits / len(gold)


def evaluate_rules(
    *,
    gold_rules: Sequence[GoldRule],
    extracted: Sequence[Rule],
    gold_verifications: Sequence[GoldVerification] = (),
    verifications: Sequence[VerificationResult] = (),
    gold_conflicts: Sequence[GoldConflictPair] = (),
    detected_conflicts: Sequence[RuleConflict] = (),
    rule_id_by_gold_id: dict[str, str] | None = None,
) -> RuleEvalReport:
    recall, precision, condition_accuracy, effect_accuracy = evaluate_extraction(gold_rules, extracted)
    verifier_agreement_rate, hallucination_rate = evaluate_verification(gold_verifications, verifications)
    conflict_detection_recall = evaluate_conflict_detection(
        gold_conflicts, detected_conflicts, rule_id_by_gold_id=rule_id_by_gold_id or {}
    )
    return RuleEvalReport(
        gold_count=len(gold_rules),
        extracted_count=len(extracted),
        recall=recall,
        precision=precision,
        condition_accuracy=condition_accuracy,
        effect_accuracy=effect_accuracy,
        verifier_agreement_rate=verifier_agreement_rate,
        hallucination_rate=hallucination_rate,
        conflict_detection_recall=conflict_detection_recall,
    )
