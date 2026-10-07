"""Lal Kitab vocabulary: the Navagraha (nine planets), twelve houses
(by Hindi/Sanskrit ordinal names, exactly as Lal Kitab texts write
them — e.g. "ashtam bhava" for the 8th house, the literal example
`ragcore.rules.domain.OntologyNormalizer`'s own docstring uses), the
twelve rashis (signs), and dignity terms — canonical English slugs on
one side, every alias/transliteration this pack recognizes on the
other.

This is deliberately a flat, hand-authored table, not a fetched or
generated one: Lal Kitab vocabulary is small and closed (9 planets, 12
houses, 12 signs, 4 dignity states), and a hand-reviewed table is more
trustworthy for something astrological rules will be matched against
than an unreviewed auto-generated one. Extend it as real source texts
surface spellings not covered here — this is not meant to be
exhaustive of every possible transliteration on first pass.
"""

from __future__ import annotations

PLANETS: tuple[str, ...] = (
    "sun", "moon", "mars", "mercury", "jupiter", "venus", "saturn", "rahu", "ketu",
)

PLANET_ALIASES: dict[str, str] = {
    "sun": "sun", "surya": "sun", "ravi": "sun",
    "moon": "moon", "chandra": "moon", "chandr": "moon", "soma": "moon",
    "mars": "mars", "mangal": "mars", "mangala": "mars", "kuja": "mars", "angarak": "mars",
    "mercury": "mercury", "budh": "mercury", "budha": "mercury",
    "jupiter": "jupiter", "guru": "jupiter", "brihaspati": "jupiter", "brihaspathi": "jupiter",
    "venus": "venus", "shukra": "venus", "shukracharya": "venus",
    "saturn": "saturn", "shani": "saturn", "shanidev": "saturn", "sani": "saturn",
    "rahu": "rahu",
    "ketu": "ketu",
}

SIGNS: tuple[str, ...] = (
    "aries", "taurus", "gemini", "cancer", "leo", "virgo",
    "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces",
)

SIGN_ALIASES: dict[str, str] = {
    "aries": "aries", "mesh": "aries", "mesha": "aries",
    "taurus": "taurus", "vrishabh": "taurus", "vrishabha": "taurus",
    "gemini": "gemini", "mithun": "gemini", "mithuna": "gemini",
    "cancer": "cancer", "kark": "cancer", "karka": "cancer",
    "leo": "leo", "simha": "leo", "sinh": "leo",
    "virgo": "virgo", "kanya": "virgo",
    "libra": "libra", "tula": "libra",
    "scorpio": "scorpio", "vrishchik": "scorpio", "vrishchika": "scorpio",
    "sagittarius": "sagittarius", "dhanu": "sagittarius",
    "capricorn": "capricorn", "makar": "capricorn", "makara": "capricorn",
    "aquarius": "aquarius", "kumbh": "aquarius", "kumbha": "aquarius",
    "pisces": "pisces", "meen": "pisces", "meena": "pisces",
}

DIGNITIES: tuple[str, ...] = ("exalted", "debilitated", "own", "neutral")

DIGNITY_ALIASES: dict[str, str] = {
    "exalted": "exalted", "uchcha": "exalted", "uch": "exalted",
    "debilitated": "debilitated", "neech": "debilitated", "neecha": "debilitated",
    "own": "own", "own sign": "own", "swagruha": "own", "swarashi": "own", "swa rashi": "own",
    "neutral": "neutral", "samanya": "neutral",
}

HOUSE_ORDINAL_ALIASES: dict[str, int] = {
    "1st": 1, "first": 1, "pratham": 1, "pehla": 1, "lagna": 1, "tanu": 1,
    "2nd": 2, "second": 2, "dwitiya": 2, "dusra": 2, "dhan": 2,
    "3rd": 3, "third": 3, "tritiya": 3, "tisra": 3, "sahaj": 3, "parakram": 3,
    "4th": 4, "fourth": 4, "chaturth": 4, "chauth": 4, "chautha": 4, "sukh": 4, "matru": 4,
    "5th": 5, "fifth": 5, "pancham": 5, "panchva": 5, "putra": 5, "santan": 5,
    "6th": 6, "sixth": 6, "shashtha": 6, "chhatha": 6, "ripu": 6, "rog": 6,
    "7th": 7, "seventh": 7, "saptam": 7, "satva": 7, "kalatra": 7, "dampatya": 7,
    "8th": 8, "eighth": 8, "ashtam": 8, "athva": 8, "mrityu": 8, "ayu": 8,
    "9th": 9, "ninth": 9, "navam": 9, "nauva": 9, "bhagya": 9, "dharma": 9,
    "10th": 10, "tenth": 10, "dasham": 10, "dasva": 10, "karma": 10, "rajya": 10,
    "11th": 11, "eleventh": 11, "ekadash": 11, "gyarvha": 11, "labh": 11, "aay": 11,
    "12th": 12, "twelfth": 12, "dwadash": 12, "barahva": 12, "vyaya": 12, "moksha": 12,
}

CONJUNCTION_WORDS: tuple[str, ...] = ("conjunct", "conjunction", "yuti", "together with")
ASPECT_WORDS: tuple[str, ...] = ("aspect", "drishti", "dekh", "sees")
HOUSE_WORDS: tuple[str, ...] = ("house", "ghar", "bhava", "bhav")
SIGN_WORDS: tuple[str, ...] = ("sign", "rashi", "rasi")
RETROGRADE_WORDS: tuple[str, ...] = ("retrograde", "vakri")
BLIND_WORDS: tuple[str, ...] = ("blind", "andha")
SLEEPING_WORDS: tuple[str, ...] = ("sleeping", "asleep", "sona", "sota")
PAKKA_GHAR_WORDS: tuple[str, ...] = ("pakka ghar", "pakka", "permanent house")
DIGNITY_WORDS: tuple[str, ...] = tuple(DIGNITY_ALIASES) + ("dignity",)
LORDSHIP_WORDS: tuple[str, ...] = ("lord", "swami", "adhipati")
