"""LalKitabPack: the DomainPack implementation (Phase 10 §7) — inert
data plus pure functions, never an extraction pipeline itself (see
docs/adr/0013). Everything in `fact_types()` is generated from the
closed, small vocabulary in `lal_kitab.vocabulary`, not hand-enumerated
one at a time — conjunction/aspect pairs especially, where hand-writing
all 36 unordered planet pairs would be repetitive and error-prone to
keep in sync with `vocabulary.PLANETS` if a tenth graha were ever added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations

from lal_kitab.ontology import LalKitabOntologyNormalizer
from lal_kitab.vocabulary import DIGNITIES, PLANETS, SIGNS
from ragcore.rules.domain import FactTypeSpec

PACK_ID = "lal_kitab_core"
PACK_VERSION = "1.0.0"


def _fact_types() -> tuple[FactTypeSpec, ...]:
    specs: list[FactTypeSpec] = []
    for planet in PLANETS:
        specs.append(FactTypeSpec(
            fact_key=f"planet.{planet}.house", value_type="int",
            description=f"The house (1-12) {planet} occupies.",
        ))
        specs.append(FactTypeSpec(
            fact_key=f"planet.{planet}.sign", value_type="enum", enum_values=SIGNS,
            description=f"The rashi (sign) {planet} occupies.",
        ))
        specs.append(FactTypeSpec(
            fact_key=f"planet.{planet}.retrograde", value_type="bool",
            description=f"Whether {planet} is retrograde (vakri).",
        ))
        specs.append(FactTypeSpec(
            fact_key=f"planet.{planet}.dignity", value_type="enum", enum_values=DIGNITIES,
            description=f"{planet}'s dignity: exalted, debilitated, own sign, or neutral.",
        ))
        specs.append(FactTypeSpec(
            fact_key=f"planet.{planet}.blind", value_type="bool",
            description=f"Whether {planet} is a \"blind\" (andha) graha in this chart.",
        ))
        specs.append(FactTypeSpec(
            fact_key=f"planet.{planet}.sleeping", value_type="bool",
            description=f"Whether {planet} is a \"sleeping\" (sona) graha in this chart.",
        ))
        specs.append(FactTypeSpec(
            fact_key=f"planet.{planet}.pakka_ghar", value_type="bool",
            description=f"Whether {planet} occupies its pakka ghar (permanent house).",
        ))

    for a, b in combinations(sorted(PLANETS), 2):
        specs.append(FactTypeSpec(
            fact_key=f"planet.{a}.conjunct.{b}", value_type="bool",
            description=f"Whether {a} and {b} are conjunct (yuti).",
        ))

    for a, b in permutations(PLANETS, 2):
        specs.append(FactTypeSpec(
            fact_key=f"planet.{a}.aspects.{b}", value_type="bool",
            description=f"Whether {a} aspects (drishti) {b} — directional.",
        ))

    for house_n in range(1, 13):
        specs.append(FactTypeSpec(
            fact_key=f"house.{house_n}.lord", value_type="enum", enum_values=PLANETS,
            description=f"Which planet rules (is lord of) house {house_n}.",
        ))

    specs.append(FactTypeSpec(
        fact_key="native.dasha_lord", value_type="enum", enum_values=PLANETS,
        description="The planet whose dasha (period) is currently running for the native.",
    ))
    specs.append(FactTypeSpec(
        fact_key="native.age", value_type="int",
        description="The native's age in years.",
    ))
    return tuple(specs)


_FACT_TYPES = _fact_types()

_EXTRACTION_SYSTEM_PROMPT = (
    "You are extracting rules from Lal Kitab, a system of Vedic astrology "
    "that gives definite, often absolute-sounding predictions and remedies "
    "based on planetary positions in a birth chart. Source text is often a "
    "mix of English and transliterated Hindi/Sanskrit astrological "
    "terminology (e.g. \"mangal\" for Mars, \"ashtam bhava\" for the 8th "
    "house, \"vakri\" for retrograde, \"pakka ghar\" for a planet's "
    "permanent/strong house, \"andha graha\" for a \"blind\" (ineffective) "
    "planet, \"sona graha\" for a \"sleeping\" (dormant) planet). Use the "
    "source text's own vocabulary in your output — do not translate Hindi "
    "terms to English or vice versa, and do not normalize spellings "
    "yourself; that is this pack's ontology's job, not yours. A rule "
    "states a condition (one or more planets' house/sign/retrograde/"
    "dignity/conjunction/aspect status, a house's lordship, or the "
    "native's age/dasha period) and an effect (and optionally a remedy) — "
    "extract only statements that actually state such a condition-effect "
    "link, not general mythology or explanation of what a planet "
    "represents."
)

_VERIFICATION_SYSTEM_PROMPT = (
    "You are verifying Lal Kitab rule extractions. Lal Kitab's own style "
    "is to state predictions in absolute, confident language even for "
    "what is actually a narrower condition than it first appears - check "
    "carefully that every stated condition (not just the effect) is "
    "actually present in the verbatim span, not assumed from general Lal "
    "Kitab convention. A span mentioning only one planet's house position "
    "does not support a rule whose conditions include a second planet's "
    "retrograde status unless that second condition is also actually "
    "stated."
)


@dataclass(frozen=True)
class LalKitabPack:
    domain: str = "lal_kitab"
    pack_id: str = PACK_ID
    pack_version: str = PACK_VERSION
    ontology: LalKitabOntologyNormalizer = field(default_factory=LalKitabOntologyNormalizer)

    def fact_types(self) -> tuple[FactTypeSpec, ...]:
        return _FACT_TYPES

    def extraction_system_prompt(self) -> str:
        return _EXTRACTION_SYSTEM_PROMPT

    def verification_system_prompt(self) -> str:
        return _VERIFICATION_SYSTEM_PROMPT
