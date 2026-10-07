"""LalKitabOntologyNormalizer: maps raw source-text mentions (English,
Hindi/Sanskrit transliterations, common alternate spellings) to the
canonical fact keys/values `ragcore.rules.evaluator` and this pack's
own `fact_types()` registry (`lal_kitab.pack`) understand.

**Fact key shapes this ontology produces** (all under domain
"lal_kitab"):
  - `planet.{canon}.house` (int 1-12)
  - `planet.{canon}.sign` (enum — one of `vocabulary.SIGNS`)
  - `planet.{canon}.retrograde` (bool)
  - `planet.{canon}.dignity` (enum — one of `vocabulary.DIGNITIES`)
  - `planet.{canon}.blind` / `.sleeping` / `.pakka_ghar` (bool — Phase
    10's own "blind/sleeping/pakka-ghar status")
  - `planet.{a}.conjunct.{b}` (bool, `a < b` alphabetically — symmetric,
    so canonicalized to one ordering regardless of mention order)
  - `planet.{a}.aspects.{b}` (bool, directional — `a` is whichever
    planet the mention names first, since Lal Kitab aspect is not
    symmetric)
  - `house.{1-12}.lord` (enum — one of `vocabulary.PLANETS`)
  - `native.dasha_lord` (enum — one of `vocabulary.PLANETS`)
  - `native.age` (int)

Word-matching is substring-based over a lowercased mention, checked in
a deliberately specific-before-general order (`pakka_ghar` before the
generic `house` check, since "pakka ghar" itself contains the
substring "ghar") — see `_FACT_TYPE_CHECKS`' own ordering comment.
Nothing here guesses: a mention this table can't resolve returns
`None`, exactly as `OntologyNormalizer`'s own contract requires,
rather than falling back to some default that would silently produce
a wrong rule.
"""

from __future__ import annotations

import re
from typing import Any

from lal_kitab.vocabulary import (
    ASPECT_WORDS,
    BLIND_WORDS,
    CONJUNCTION_WORDS,
    DIGNITY_ALIASES,
    DIGNITY_WORDS,
    HOUSE_ORDINAL_ALIASES,
    HOUSE_WORDS,
    LORDSHIP_WORDS,
    PAKKA_GHAR_WORDS,
    PLANET_ALIASES,
    RETROGRADE_WORDS,
    SIGN_ALIASES,
    SIGN_WORDS,
    SLEEPING_WORDS,
)

_TRUE_WORDS = ("yes", "true", "haan", "hai")
_FALSE_WORDS = ("no", "false", "nahi", "nahin")


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(w in text for w in words)


def _find_alias(text: str, aliases: dict[str, str]) -> str | None:
    """Longest-alias-first substring match — "swagruha" must be
    checked before any single-letter-shorter false positive, and e.g.
    "mangala" before "mangal" would matter if "mangal" were ever a
    prefix-colliding false match (it isn't here, but the ordering
    discipline is kept uniform everywhere this helper is used)."""
    for alias in sorted(aliases, key=len, reverse=True):
        if alias in text:
            return aliases[alias]
    return None


def _find_planets_in_order(text: str) -> list[str]:
    """Every distinct canonical planet mentioned, in the order its
    *first* alias occurrence appears in the text — used to resolve
    conjunction/aspect mentions naming two planets at once."""
    positions: list[tuple[int, str]] = []
    for alias, canon in PLANET_ALIASES.items():
        match = re.search(rf"\b{re.escape(alias)}\b", text)
        if match:
            positions.append((match.start(), canon))
    positions.sort()
    ordered: list[str] = []
    for _, canon in positions:
        if canon not in ordered:
            ordered.append(canon)
    return ordered


def _find_house_number(text: str) -> int | None:
    for alias in sorted(HOUSE_ORDINAL_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return HOUSE_ORDINAL_ALIASES[alias]
    bare = re.search(r"\b(\d{1,2})\b", text)
    if bare:
        n = int(bare.group(1))
        if 1 <= n <= 12:
            return n
    return None


# Checked in this order deliberately: pakka_ghar before house (the
# phrase "pakka ghar" itself contains "ghar"), everything else is
# mutually exclusive vocabulary.
_FACT_TYPE_CHECKS: tuple[tuple[tuple[str, ...], str], ...] = (
    (PAKKA_GHAR_WORDS, "pakka_ghar"),
    (BLIND_WORDS, "blind"),
    (SLEEPING_WORDS, "sleeping"),
    (RETROGRADE_WORDS, "retrograde"),
    (DIGNITY_WORDS, "dignity"),
    (HOUSE_WORDS, "house"),
    (SIGN_WORDS, "sign"),
)


class LalKitabOntologyNormalizer:
    def normalize_fact_key(self, raw_mention: str) -> str | None:
        text = raw_mention.lower().strip()
        if not text:
            return None

        # Checked before any planet-specific branch: a dasha mention
        # (e.g. "mangal dasha") names a planet too, but means something
        # else entirely (native.dasha_lord, not a planet.* fact) - it
        # must not fall through to the planet-house/sign/... checks
        # below just because a planet happens to also be present.
        if "dasha" in text or "mahadasha" in text:
            return "native.dasha_lord"
        if "age" in text or "umar" in text:
            return "native.age"

        if _contains_any(text, LORDSHIP_WORDS):
            house_n = _find_house_number(text)
            return f"house.{house_n}.lord" if house_n is not None else None

        planets = _find_planets_in_order(text)

        if not planets:
            return None

        if _contains_any(text, CONJUNCTION_WORDS):
            if len(planets) < 2:
                return None
            a, b = sorted(planets[:2])
            return f"planet.{a}.conjunct.{b}"

        if _contains_any(text, ASPECT_WORDS):
            if len(planets) < 2:
                return None
            return f"planet.{planets[0]}.aspects.{planets[1]}"

        planet = planets[0]
        for words, fact_type in _FACT_TYPE_CHECKS:
            if _contains_any(text, words):
                return f"planet.{planet}.{fact_type}"
        return None

    def normalize_fact_value(self, fact_key: str, raw_mention: str) -> Any | None:
        text = raw_mention.lower().strip()
        if not text:
            return None

        if fact_key.endswith(".house"):
            return _find_house_number(text)
        if fact_key.startswith("house.") and fact_key.endswith(".lord"):
            return _find_alias(text, PLANET_ALIASES)
        if fact_key == "native.dasha_lord":
            return _find_alias(text, PLANET_ALIASES)
        if fact_key == "native.age":
            bare = re.search(r"\b(\d{1,3})\b", text)
            return int(bare.group(1)) if bare else None
        if fact_key.endswith(".sign"):
            return _find_alias(text, SIGN_ALIASES)
        if fact_key.endswith(".dignity"):
            return _find_alias(text, DIGNITY_ALIASES)
        if fact_key.endswith((".retrograde", ".blind", ".sleeping", ".pakka_ghar")) or ".conjunct." in fact_key or ".aspects." in fact_key:
            if _contains_any(text, _TRUE_WORDS):
                return True
            if _contains_any(text, _FALSE_WORDS):
                return False
            return None
        return None
