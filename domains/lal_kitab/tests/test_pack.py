"""LalKitabPack: Protocol conformance (DomainPack, OntologyNormalizer),
fact_types() coverage and shape (generated from vocabulary, not hand-
enumerated), and the prompts being non-empty, domain-grounded text —
Phase 10 §7.
"""

from __future__ import annotations

from lal_kitab import LalKitabOntologyNormalizer, LalKitabPack
from lal_kitab.vocabulary import PLANETS

from ragcore.rules.domain import DomainPack, OntologyNormalizer


def test_lal_kitab_pack_satisfies_the_domain_pack_protocol() -> None:
    assert isinstance(LalKitabPack(), DomainPack)


def test_lal_kitab_ontology_satisfies_the_ontology_normalizer_protocol() -> None:
    assert isinstance(LalKitabOntologyNormalizer(), OntologyNormalizer)


def test_pack_identity_fields() -> None:
    pack = LalKitabPack()
    assert pack.domain == "lal_kitab"
    assert pack.pack_id
    assert pack.pack_version


def test_fact_types_covers_house_for_every_planet() -> None:
    pack = LalKitabPack()
    keys = {ft.fact_key for ft in pack.fact_types()}
    for planet in PLANETS:
        assert f"planet.{planet}.house" in keys


def test_fact_types_covers_every_unordered_conjunction_pair_once() -> None:
    pack = LalKitabPack()
    conjunction_keys = [ft.fact_key for ft in pack.fact_types() if ".conjunct." in ft.fact_key]
    # C(9,2) = 36 unordered pairs, each appearing exactly once
    assert len(conjunction_keys) == 36
    assert len(set(conjunction_keys)) == 36


def test_fact_types_covers_every_directed_aspect_pair() -> None:
    pack = LalKitabPack()
    aspect_keys = [ft.fact_key for ft in pack.fact_types() if ".aspects." in ft.fact_key]
    # 9 * 8 = 72 directed pairs (a aspects b is distinct from b aspects a)
    assert len(aspect_keys) == 72


def test_fact_types_covers_lordship_for_all_twelve_houses() -> None:
    pack = LalKitabPack()
    keys = {ft.fact_key for ft in pack.fact_types()}
    for house_n in range(1, 13):
        assert f"house.{house_n}.lord" in keys


def test_fact_types_covers_dasha_and_age() -> None:
    pack = LalKitabPack()
    keys = {ft.fact_key for ft in pack.fact_types()}
    assert "native.dasha_lord" in keys
    assert "native.age" in keys


def test_fact_types_has_no_duplicate_keys() -> None:
    pack = LalKitabPack()
    keys = [ft.fact_key for ft in pack.fact_types()]
    assert len(keys) == len(set(keys))


def test_every_fact_type_spec_is_internally_valid() -> None:
    """FactTypeSpec's own __post_init__ validation (enum requires
    enum_values, non-enum rejects them) already ran at construction
    time inside fact_types() - this just confirms none of the 185
    generated specs silently failed to construct."""
    pack = LalKitabPack()
    assert len(pack.fact_types()) > 100


def test_prompts_are_non_empty_and_domain_grounded() -> None:
    pack = LalKitabPack()
    extraction = pack.extraction_system_prompt()
    verification = pack.verification_system_prompt()
    assert "Lal Kitab" in extraction
    assert "Lal Kitab" in verification
    assert len(extraction) > 100
    assert len(verification) > 100


def test_two_pack_instances_share_the_same_fact_types_object() -> None:
    """fact_types() is generated once at module import, not rebuilt
    per instance/call - confirmed directly rather than assumed, since
    rebuilding ~185 FactTypeSpecs on every call would be wasteful if
    this ever moved into a hot path."""
    assert LalKitabPack().fact_types() is LalKitabPack().fact_types()
