"""Shared test helper, not a test module itself (no test_ prefix).
Builds a Rule from raw Lal Kitab mentions via the real
LalKitabOntologyNormalizer — mirrors ragcore.rules.claude's
_build_conditions logic by hand, without going through the LLM-backed
extractor, so a test can prove the ontology's output is directly
usable by the schema with no adapter code.
"""

from __future__ import annotations

from lal_kitab import LalKitabPack

from ragcore.rules.types import (
    ConditionGroup,
    Effect,
    FactPredicate,
    Rule,
    rule_content_hash,
)


def build_rule_from_mentions(
    pack: LalKitabPack,
    *,
    book_id: str,
    chunk_id: str,
    verbatim_span: str,
    mentions,
    effect: Effect,
    rule_type: str = "HARD",
) -> Rule:
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
        book_id=book_id, page_start=1, chunk_ids=(chunk_id,),
        verbatim_span=verbatim_span, language="hi-en", normalized_statement=verbatim_span,
        conditions=conditions, effect=effect, rule_type=rule_type, confidence=0.95,
        extraction_model="test", extraction_version="v1", created_at="now",
        review_status="verified",
    )
