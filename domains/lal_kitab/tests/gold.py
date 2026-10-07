"""Hand-labeled Lal Kitab gold set (Phase 10 §6: "Hand-labelled gold
set for 3 chapters") plus the matching simulated extraction output —
not a test module itself (no test_ prefix); test_eval_gold.py is the
regression gate that runs ragcore.rules.eval against this data.

**This is a hand-authored synthetic fixture, not excerpted from any
real published book** — the same "synthetic fixture, not a real
source" discipline `tests/integration/`'s own "8 synthetic books"
already follows in the main ragcore repo, for the same reason: a
regression gate needs a fixed, reusable, freely-distributable text
every run compares against, and using real copyrighted Lal Kitab text
would be neither. Each "chapter" below is a short passage written to
be *representative* of genuine Lal Kitab rule style and vocabulary
(the same planets/houses/conditions/effects `lal_kitab.vocabulary`
documents), not an actual translation of anything published.

Three "chapters" (source chunks), matching the spec's own "3
chapters": chapter 1 covers a simple HARD rule, chapter 2 an EXCEPTION
overriding it, chapter 3 an unrelated TENDENCY rule plus one
deliberately *not* extracted (a missed rule, kept in the gold set, so
recall on this fixture isn't trivially 1.0 by construction — a
regression gate that can't distinguish "extraction found everything"
from "extraction found nothing" wouldn't be testing anything).
"""

from __future__ import annotations

from helpers import build_rule_from_mentions
from lal_kitab import LalKitabPack

from ragcore.rules.base import VerificationResult
from ragcore.rules.eval import GoldConflictPair, GoldRule, GoldVerification
from ragcore.rules.types import Effect, Rule

BOOK_ID = "lal_kitab_gold_book"

CH1_CHUNK = f"{BOOK_ID}:ch1"
CH2_CHUNK = f"{BOOK_ID}:ch2"
CH3_CHUNK = f"{BOOK_ID}:ch3"

CH1_TEXT = (
    "Mangal ashtam bhava mein ho to swasthya kasht hota hai — "
    "Mars in the 8th house causes health troubles."
)
CH2_TEXT = (
    "Lekin Mangal vakri ho to yeh kasht nahi hota — "
    "but if Mars is retrograde, this trouble does not occur."
)
CH3_TEXT = (
    "Shani saptam bhava mein ho to vivah mein deri ho sakti hai — "
    "Saturn in the 7th house may delay marriage. "
    "Guru dasham bhava mein ho to karmakshetra mein samman milta hai — "
    "Jupiter in the 10th house tends to bring respect in one's career."
)


def _gold_rules(pack: LalKitabPack) -> list[GoldRule]:
    ch1 = build_rule_from_mentions(
        pack, book_id=BOOK_ID, chunk_id=CH1_CHUNK, verbatim_span=CH1_TEXT,
        mentions=[("mangal house", "eq", "ashtam bhava")],
        effect=Effect(category="health", polarity="negative"),
    )
    ch2 = build_rule_from_mentions(
        pack, book_id=BOOK_ID, chunk_id=CH2_CHUNK, verbatim_span=CH2_TEXT,
        mentions=[("mangal house", "eq", "ashtam bhava"), ("mangal vakri", "exists", None)],
        effect=Effect(category="health", polarity="positive"),
        rule_type="EXCEPTION",
    )
    ch3_marriage = build_rule_from_mentions(
        pack, book_id=BOOK_ID, chunk_id=CH3_CHUNK, verbatim_span=CH3_TEXT,
        mentions=[("shani house", "eq", "saptam bhava")],
        effect=Effect(category="marriage", polarity="negative"),
        rule_type="TENDENCY",
    )
    ch3_career = build_rule_from_mentions(
        pack, book_id=BOOK_ID, chunk_id=CH3_CHUNK, verbatim_span=CH3_TEXT,
        mentions=[("guru house", "eq", "dasham bhava")],
        effect=Effect(category="career", polarity="positive"),
        rule_type="TENDENCY",
    )
    return [
        GoldRule(gold_id="gold_ch1_mars_8th", chunk_id=CH1_CHUNK, conditions=ch1.conditions, effect=ch1.effect),
        GoldRule(gold_id="gold_ch2_exception", chunk_id=CH2_CHUNK, conditions=ch2.conditions, effect=ch2.effect),
        GoldRule(gold_id="gold_ch3_marriage", chunk_id=CH3_CHUNK, conditions=ch3_marriage.conditions, effect=ch3_marriage.effect),
        GoldRule(gold_id="gold_ch3_career", chunk_id=CH3_CHUNK, conditions=ch3_career.conditions, effect=ch3_career.effect),
    ], {
        "gold_ch1_mars_8th": ch1,
        "gold_ch2_exception": ch2,
        "gold_ch3_marriage": ch3_marriage,
        "gold_ch3_career": ch3_career,
    }


def build_gold_fixture(pack: LalKitabPack) -> tuple[
    list[GoldRule], list[Rule], list[GoldVerification], list[VerificationResult],
    list[GoldConflictPair], dict[str, str],
]:
    """Returns (gold_rules, simulated_extraction, gold_verifications,
    verifications, gold_conflicts, rule_id_by_gold_id) — a self-
    consistent fixture where the "simulated extraction" is what a
    correctly-functioning pipeline SHOULD produce for this fixed text,
    deliberately missing `gold_ch3_career` (the "missed rule" this
    module's own docstring describes) and including one hallucinated
    rule the gold verification labels as unsupported."""
    gold_rules, rules_by_gold_id = _gold_rules(pack)

    # The simulated extraction "misses" gold_ch3_career entirely (nothing
    # extracted for it) and adds one hallucinated rule not in the gold set.
    hallucinated = build_rule_from_mentions(
        pack, book_id=BOOK_ID, chunk_id=CH3_CHUNK,
        verbatim_span="(a hallucinated rule not actually stated in the text)",
        mentions=[("shani vakri", "exists", None)],
        effect=Effect(category="wealth", polarity="negative"),
    )
    extracted = [
        rules_by_gold_id["gold_ch1_mars_8th"],
        rules_by_gold_id["gold_ch2_exception"],
        rules_by_gold_id["gold_ch3_marriage"],
        hallucinated,
    ]

    gold_verifications = [
        GoldVerification(rule_id=rules_by_gold_id["gold_ch1_mars_8th"].rule_id, human_supported=True),
        GoldVerification(rule_id=rules_by_gold_id["gold_ch2_exception"].rule_id, human_supported=True),
        GoldVerification(rule_id=rules_by_gold_id["gold_ch3_marriage"].rule_id, human_supported=True),
        GoldVerification(rule_id=hallucinated.rule_id, human_supported=False),
    ]
    # A correctly-functioning verifier should agree with every one of these.
    verifications = [
        VerificationResult(rule_id=g.rule_id, supported=g.human_supported, reason="matches gold")
        for g in gold_verifications
    ]

    gold_conflicts = [
        GoldConflictPair(gold_id_a="gold_ch1_mars_8th", gold_id_b="gold_ch2_exception", conflict_type="exception"),
    ]
    rule_id_by_gold_id = {gold_id: rule.rule_id for gold_id, rule in rules_by_gold_id.items()}

    return gold_rules, extracted, gold_verifications, verifications, gold_conflicts, rule_id_by_gold_id
