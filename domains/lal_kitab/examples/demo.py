"""Demonstrates the Lal Kitab domain pack driving the fully generic
ragcore.rules layer end to end: build rules from raw Lal Kitab
mentions via the real ontology (no LLM call — the same hand-built-Rule
technique domains/lal_kitab/tests/test_integration.py uses, standing
in for ClaudeRuleExtractor), detect a general-vs-exception conflict,
store it, then serve a chart query and show precedence resolution in
action.

No API calls, no network, no sample corpus required — every Rule here
is built directly from the ontology's own normalize_fact_key/
normalize_fact_value, which is exactly what ClaudeRuleExtractor does
internally (see ragcore/rules/claude.py's _build_conditions) once a
real extraction call has produced the (fact_mention, operator,
value_mention) triples; this script just skips the LLM step and
supplies those triples directly, in the source's own mixed Hindi/
English vocabulary, to prove the ontology path works without needing
Claude credentials to run.

Usage (from domains/lal_kitab/, with ragcore[generation] and this
package installed):
    python examples/demo.py
"""

from __future__ import annotations

from lal_kitab import LalKitabPack

from ragcore.rules.consistency import detect_conflicts
from ragcore.rules.evaluator import evaluate_condition
from ragcore.rules.serving import match_with_provenance
from ragcore.rules.sqlite_store import SqliteRuleStore
from ragcore.rules.types import (
    ConditionGroup,
    Effect,
    FactPredicate,
    Rule,
    rule_content_hash,
)


def build_rule_from_mentions(pack, *, book_id, chunk_id, verbatim_span, mentions, effect, rule_type="HARD"):
    predicates = []
    for fact_mention, operator, value_mention in mentions:
        fact_key = pack.ontology.normalize_fact_key(fact_mention)
        assert fact_key is not None, f"couldn't normalize {fact_mention!r}"
        if operator == "exists":
            predicates.append(FactPredicate(fact=fact_key, operator="exists"))
        else:
            value = pack.ontology.normalize_fact_value(fact_key, value_mention)
            assert value is not None, f"couldn't normalize value {value_mention!r} for {fact_key!r}"
            predicates.append(FactPredicate(fact=fact_key, operator=operator, value=value))
    conditions = predicates[0] if len(predicates) == 1 else ConditionGroup(op="and", children=predicates)
    rule_id = rule_content_hash(
        domain=pack.domain, book_id=book_id, verbatim_span=verbatim_span,
        conditions=conditions, effect=effect,
    )
    return Rule(
        rule_id=rule_id, domain=pack.domain, pack_id=pack.pack_id, pack_version=pack.pack_version,
        book_id=book_id, page_start=1, chunk_ids=(chunk_id,), verbatim_span=verbatim_span,
        language="hi-en", normalized_statement=verbatim_span, conditions=conditions, effect=effect,
        rule_type=rule_type, confidence=0.95, extraction_model="demo", extraction_version="v1",
        created_at="now", review_status="verified",
    )


def main() -> None:
    pack = LalKitabPack()
    print(f"Loaded domain pack: {pack.domain} ({pack.pack_id} v{pack.pack_version}), "
          f"{len(pack.fact_types())} registered fact types.\n")

    general = build_rule_from_mentions(
        pack, book_id="lal_kitab_book", chunk_id="lal_kitab_book:ch3",
        verbatim_span="Mangal ashtam bhava mein ho to swasthya kasht hota hai.",
        mentions=[("mangal house", "eq", "ashtam bhava")],
        effect=Effect(category="health", polarity="negative"),
    )
    exception = build_rule_from_mentions(
        pack, book_id="lal_kitab_book", chunk_id="lal_kitab_book:ch4",
        verbatim_span="Lekin Mangal vakri ho to yeh kasht nahi hota.",
        mentions=[("mangal house", "eq", "ashtam bhava"), ("mangal vakri", "exists", None)],
        effect=Effect(category="health", polarity="positive"),
        rule_type="EXCEPTION",
    )
    print("Built 2 rules from raw Lal Kitab mentions:")
    for rule in (general, exception):
        print(f"  [{rule.rule_type:>9}] {rule.rule_id}: {rule.verbatim_span!r}")
        print(f"             conditions = {rule.conditions.model_dump()}")

    store = SqliteRuleStore(":memory:")
    store.add_rules([general, exception])

    conflicts = detect_conflicts([general, exception])
    store.put_conflicts(conflicts)
    print(f"\nDetected {len(conflicts)} conflict(s):")
    for c in conflicts:
        print(f"  {c.conflict_type}: {c.rule_ids} -> winner: {c.winning_rule_id}")

    chart_with_retrograde_mars = {"planet.mars.house": 8, "planet.mars.retrograde": True}
    chart_without_retrograde = {"planet.mars.house": 8}

    print("\nChart A (Mars in 8th house, retrograde):")
    for rule in (general, exception):
        print(f"  {rule.rule_type:>9} holds: {evaluate_condition(rule.conditions, chart_with_retrograde_mars)}")
    result_a = match_with_provenance([general, exception], chart_with_retrograde_mars, store=store)
    print(f"  Effective rule(s): {[r.rule_id for r in result_a.effective]}  <- the EXCEPTION wins")

    print("\nChart B (Mars in 8th house, NOT retrograde):")
    for rule in (general, exception):
        print(f"  {rule.rule_type:>9} holds: {evaluate_condition(rule.conditions, chart_without_retrograde)}")
    result_b = match_with_provenance([general, exception], chart_without_retrograde, store=store)
    print(f"  Effective rule(s): {[r.rule_id for r in result_b.effective]}  <- only the general rule applies")


if __name__ == "__main__":
    main()
