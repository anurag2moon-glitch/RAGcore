"""End-to-end proof that LalKitabPack drives the fully generic
ragcore.rules layer (schema, evaluator, consistency, serving) with no
core changes — the concrete claim docs/adr/0013 makes. Builds a Rule
from ontology-normalized Lal Kitab mentions (not hand-picked canonical
facts) and evaluates it against a realistic chart fact set, exactly
the shape ragcore.rules.consistency/serving expect, with zero
Lal-Kitab-specific code anywhere in ragcore itself.
"""

from __future__ import annotations

from helpers import build_rule_from_mentions
from lal_kitab import LalKitabPack

from ragcore.rules.consistency import detect_conflicts
from ragcore.rules.evaluator import evaluate_condition, match_rules, validate_facts
from ragcore.rules.serving import match_with_provenance
from ragcore.rules.sqlite_store import SqliteRuleStore
from ragcore.rules.types import Effect


def _build_rule_from_mentions(pack: LalKitabPack, *, verbatim_span: str, mentions, effect: Effect):
    return build_rule_from_mentions(
        pack, book_id="lal_kitab_ch1", chunk_id="lal_kitab_ch1:5",
        verbatim_span=verbatim_span, mentions=mentions, effect=effect,
    )


def test_a_rule_built_from_raw_lal_kitab_mentions_matches_a_realistic_chart() -> None:
    pack = LalKitabPack()
    rule = _build_rule_from_mentions(
        pack,
        verbatim_span="Mangal ashtam bhava mein ho to swasthya kasht hota hai.",
        mentions=[("mangal house", "eq", "ashtam bhava")],
        effect=Effect(category="health", polarity="negative"),
    )

    chart = {"planet.mars.house": 8, "planet.saturn.house": 1}
    assert evaluate_condition(rule.conditions, chart) is True
    assert match_rules([rule], chart) == [rule]

    # a different chart where Mars is NOT in the 8th house doesn't match
    other_chart = {"planet.mars.house": 3}
    assert evaluate_condition(rule.conditions, other_chart) is False


def test_validate_facts_catches_a_malformed_chart_before_matching() -> None:
    pack = LalKitabPack()
    errors = validate_facts({"planet.mars.house": "eighth, not a number"}, pack.fact_types())
    assert len(errors) == 1
    assert "planet.mars.house" in errors[0]


def test_an_exception_rule_built_from_mentions_overrides_the_general_rule() -> None:
    pack = LalKitabPack()
    general = _build_rule_from_mentions(
        pack, verbatim_span="Mangal ashtam bhava mein kasht deta hai.",
        mentions=[("mangal house", "eq", "ashtam bhava")],
        effect=Effect(category="health", polarity="negative"),
    )
    exception = _build_rule_from_mentions(
        pack, verbatim_span="Lekin Mangal vakri ho to yeh kasht nahi hota.",
        mentions=[("mangal house", "eq", "ashtam bhava"), ("mangal vakri", "exists", None)],
        effect=Effect(category="health", polarity="positive"),
    )
    exception = exception.model_copy(update={"rule_type": "EXCEPTION"})

    store = SqliteRuleStore(":memory:")
    store.put_conflicts(detect_conflicts([general, exception]))

    chart = {"planet.mars.house": 8, "planet.mars.retrograde": True}
    result = match_with_provenance([general, exception], chart, store=store)

    assert {r.rule_id for r in result.effective} == {exception.rule_id}
