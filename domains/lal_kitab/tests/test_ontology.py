"""LalKitabOntologyNormalizer: every alias table is exercised end to
end (not just spot-checked), the compound conjunction/aspect/lordship/
dasha parsing, and the "mangal dasha" priority-ordering bug a direct
execution caught before this suite was written (a dasha mention also
names a planet, so it must not fall through to a planet.* fact check).
"""

from __future__ import annotations

import pytest
from lal_kitab.ontology import LalKitabOntologyNormalizer
from lal_kitab.vocabulary import (
    DIGNITY_ALIASES,
    HOUSE_ORDINAL_ALIASES,
    PLANET_ALIASES,
    SIGN_ALIASES,
)


@pytest.fixture
def ontology() -> LalKitabOntologyNormalizer:
    return LalKitabOntologyNormalizer()


# -- the literal example from ragcore.rules.domain's own docstring ------


def test_ashtam_bhava_normalizes_to_house_8(ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_value("planet.mars.house", "ashtam bhava") == 8


# -- every planet alias, exhaustively -------------------------------------


@pytest.mark.parametrize(("alias", "canon"), sorted(PLANET_ALIASES.items()))
def test_every_planet_alias_resolves_a_house_mention(alias: str, canon: str, ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_key(f"{alias} house") == f"planet.{canon}.house"


@pytest.mark.parametrize(("alias", "canon"), sorted(PLANET_ALIASES.items()))
def test_every_planet_alias_resolves_as_a_lordship_value(alias: str, canon: str, ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_value("house.7.lord", alias) == canon


# -- every sign alias, exhaustively ---------------------------------------


@pytest.mark.parametrize(("alias", "canon"), sorted(SIGN_ALIASES.items()))
def test_every_sign_alias_resolves(alias: str, canon: str, ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_value("planet.mars.sign", alias) == canon


# -- every house ordinal alias, exhaustively -------------------------------


@pytest.mark.parametrize(("alias", "n"), sorted(HOUSE_ORDINAL_ALIASES.items()))
def test_every_house_ordinal_alias_resolves(alias: str, n: int, ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_value("planet.mars.house", alias) == n


# -- every dignity alias, exhaustively --------------------------------


@pytest.mark.parametrize(("alias", "canon"), sorted(DIGNITY_ALIASES.items()))
def test_every_dignity_alias_resolves(alias: str, canon: str, ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_value("planet.mars.dignity", alias) == canon


# -- fact-type disambiguation (single planet) ------------------------


def test_pakka_ghar_is_not_mistaken_for_a_generic_house_mention(ontology: LalKitabOntologyNormalizer) -> None:
    """"pakka ghar" itself contains the substring "ghar" (a HOUSE_WORDS
    trigger) - pakka_ghar must still win, not the generic house check."""
    assert ontology.normalize_fact_key("guru pakka ghar") == "planet.jupiter.pakka_ghar"


def test_blind_planet_mention(ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_key("shani andha") == "planet.saturn.blind"


def test_sleeping_planet_mention(ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_key("guru sona") == "planet.jupiter.sleeping"


def test_retrograde_mention(ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_key("shani vakri") == "planet.saturn.retrograde"


def test_dignity_mention_via_the_value_word_itself(ontology: LalKitabOntologyNormalizer) -> None:
    """The fact-type keyword set for dignity includes the enum value
    words themselves (exalted/uchcha/debilitated/neech/...), since a
    source rarely uses the literal word "dignity"."""
    assert ontology.normalize_fact_key("mangal exalted") == "planet.mars.dignity"
    assert ontology.normalize_fact_value("planet.mars.dignity", "exalted") == "exalted"


# -- conjunction / aspect (two-planet, compound mentions) ------------------


def test_conjunction_is_canonicalized_regardless_of_mention_order(ontology: LalKitabOntologyNormalizer) -> None:
    forward = ontology.normalize_fact_key("mangal shani yuti")
    backward = ontology.normalize_fact_key("shani mangal conjunction")
    assert forward == backward == "planet.mars.conjunct.saturn"


def test_aspect_is_directional_first_named_planet_is_the_subject(ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_key("mangal drishti shani") == "planet.mars.aspects.saturn"
    assert ontology.normalize_fact_key("shani drishti mangal") == "planet.saturn.aspects.mars"


def test_conjunction_with_only_one_planet_named_is_unresolvable(ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_key("mangal yuti") is None


def test_conjunction_and_aspect_value_words(ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_value("planet.mars.conjunct.saturn", "haan") is True
    assert ontology.normalize_fact_value("planet.mars.aspects.saturn", "nahi") is False
    assert ontology.normalize_fact_value("planet.mars.aspects.saturn", "unclear") is None


# -- lordship (house -> planet) -----------------------------------------


def test_lordship_mention_resolves_the_house_number_as_the_key(ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_key("saptam bhava swami") == "house.7.lord"
    assert ontology.normalize_fact_key("lord of the 10th house") == "house.10.lord"


def test_lordship_without_a_resolvable_house_number_is_unresolvable(ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_key("some vague lord mention") is None


# -- dasha / age (no planet-specific fact key) --------------------------


def test_dasha_mention_resolves_to_native_dasha_lord_not_a_planet_fact(ontology: LalKitabOntologyNormalizer) -> None:
    """Regression test for a real bug caught by direct execution before
    this suite existed: "mangal dasha" also names a planet (Mars), so
    an earlier version of the priority ordering let it fall through to
    the planet.*.house/sign/... checks (none matched "dasha", so it
    silently returned None) instead of recognizing the dasha mention
    first."""
    assert ontology.normalize_fact_key("mangal dasha") == "native.dasha_lord"
    assert ontology.normalize_fact_value("native.dasha_lord", "mangal") == "mars"


def test_age_mention(ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_key("age 32 years") == "native.age"
    assert ontology.normalize_fact_value("native.age", "32 years") == 32


# -- unresolvable mentions return None, never a guess ------------------


def test_unrecognized_mention_returns_none(ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_key("completely unrelated gibberish") is None


def test_empty_mention_returns_none(ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_key("") is None
    assert ontology.normalize_fact_value("planet.mars.house", "") is None


def test_unrecognized_value_for_a_known_key_returns_none(ontology: LalKitabOntologyNormalizer) -> None:
    assert ontology.normalize_fact_value("planet.mars.sign", "not a real sign") is None
