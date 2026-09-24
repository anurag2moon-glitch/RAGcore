"""Deterministic synthetic prose generator.

Produces readable, grammatically well-formed (if not exactly literary)
English paragraphs from a fixed set of themed vocabularies and sentence
templates, driven entirely by a seeded `random.Random` — the same seed
always produces byte-identical text, which is what lets the generated
library be cached on disk and regenerated identically anywhere.

Scope note: this generator exists to stress-test ragcore's *pipeline*
(loaders, chunker, stores, graph, retrieval) at realistic document scale
and structural variety — not to evaluate embedding quality. It is not a
language model; it will never produce prose a person would enjoy reading.
Needle phrases (`make_needle`) are deliberately rare, multi-word, thematic
phrases with a unique embedded marker token, so a query built from the same
words has an exactly-known single correct chunk — that is what makes
Recall@k/nDCG/MRR in `tests/integration/eval` meaningful against a lexical
(not semantic) embedder: see `tests/integration/doubles.DeterministicEmbedder`.
"""

from __future__ import annotations

import random
import textwrap
from dataclasses import dataclass

GENERATOR_VERSION = 1

# --------------------------------------------------------------------------
# Themed vocabularies. Each theme is deliberately lexically distinct from
# the others (different nouns/verbs/places) so DeterministicEmbedder's
# bag-of-words vectors actually separate books by subject, the same way a
# real semantic embedder would separate books by topic.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    name: str
    subjects: tuple[str, ...]
    verbs: tuple[str, ...]
    objects: tuple[str, ...]
    places: tuple[str, ...]
    adjectives: tuple[str, ...]
    adverbs: tuple[str, ...]
    artifacts: tuple[str, ...]


THEMES: dict[str, Theme] = {
    "nautical": Theme(
        name="nautical",
        subjects=("the captain", "the first mate", "a young deckhand", "the harbor pilot",
                   "the ship's cook", "the navigator", "an old sailor", "the quartermaster"),
        verbs=("charted", "hauled", "lashed", "sighted", "steered", "mended", "abandoned",
               "trusted", "questioned", "signaled to", "wrestled with", "studied"),
        objects=("the rigging", "a torn sail", "the ballast", "the tide charts", "the anchor chain",
                  "a coil of rope", "the ship's log", "the wheel", "a battered compass"),
        places=("the windward deck", "the harbor mouth", "the coral shoals", "the fog bank",
                 "the southern strait", "the quarterdeck", "the lower hold", "the open swell"),
        adjectives=("weathered", "restless", "brine-soaked", "unyielding", "half-forgotten",
                     "storm-battered", "watchful", "seasoned"),
        adverbs=("grimly", "steadily", "warily", "swiftly", "without a word", "at last"),
        artifacts=("a brass astrolabe", "an obsidian compass", "a salt-cured ledger",
                    "a tarnished sextant", "a whale-ivory whistle", "a lead-lined chest"),
    ),
    "botanical": Theme(
        name="botanical",
        subjects=("the botanist", "her assistant", "the expedition's cartographer", "a local guide",
                   "the young apprentice", "the herbalist", "the survey leader", "an itinerant collector"),
        verbs=("catalogued", "pressed", "sketched", "transplanted", "misidentified", "argued over",
               "carried", "abandoned", "rediscovered", "labeled", "pruned", "studied"),
        objects=("the specimen case", "a pressed fern", "the field notebook", "a root cutting",
                  "the seed vault", "a flowering vine", "the herbarium sheet", "a cluster of spores"),
        places=("the upper canopy", "the riverbank terrace", "the limestone ridge", "the fern gully",
                 "the greenhouse annex", "the switchback trail", "the marsh flats", "the orchard wall"),
        adjectives=("meticulous", "sun-bleached", "thorn-scarred", "patient", "half-catalogued",
                     "overgrown", "fastidious", "weather-worn"),
        adverbs=("carefully", "reluctantly", "methodically", "at dawn", "without complaint", "finally"),
        artifacts=("a cinnabar seal", "a brass-hinged specimen box", "a mildewed field journal",
                    "an ivory-handled trowel", "a wax-sealed vial", "a coppery seed tin"),
    ),
    "detective": Theme(
        name="detective",
        subjects=("the inspector", "her partner", "the young clerk", "a nervous witness",
                   "the coroner", "the landlady", "the retired sergeant", "the newspaper man"),
        verbs=("questioned", "suspected", "dismissed", "cross-examined", "shadowed", "confronted",
               "reconsidered", "distrusted", "tracked", "interrupted", "outmaneuvered", "cornered"),
        objects=("the pawn ticket", "a torn telegram", "the missing ledger", "a set of forged keys",
                  "the bloodied cravat", "a half-burnt letter", "the hidden safe", "a muddied boot print"),
        places=("the fog-choked alley", "the tenement stairwell", "the counting house", "the rail platform",
                 "the pawnbroker's back room", "the courthouse steps", "the boarding house parlor", "the dockside tavern"),
        adjectives=("suspicious", "unflappable", "tight-lipped", "methodical", "world-weary",
                     "sharp-eyed", "unrepentant", "quietly desperate"),
        adverbs=("coolly", "without flinching", "at length", "grudgingly", "abruptly", "under his breath"),
        artifacts=("a monogrammed cufflink", "a wax-sealed envelope", "a pawned pocket watch",
                    "a bloodstained glove", "an unsigned confession", "a counterfeit banknote"),
    ),
    "alpine": Theme(
        name="alpine",
        subjects=("the guide", "the youngest climber", "the expedition doctor", "a local herdsman",
                   "the porter", "the surveyor", "an aging mountaineer", "the weather watcher"),
        verbs=("scaled", "roped", "abandoned", "misjudged", "traversed", "carried", "sighted",
               "warned", "trusted", "doubted", "pushed through", "retreated from"),
        objects=("the ice axe", "a frayed rope", "the supply cache", "the weather glass",
                  "a coil of pitons", "the base-camp tent", "a spare oxygen flask", "the summit register"),
        places=("the northern couloir", "the glacier's edge", "the wind-scoured ridge", "the base camp",
                 "the crevasse field", "the tree line", "the col below the summit", "the moraine slope"),
        adjectives=("frostbitten", "unrelenting", "iron-willed", "half-frozen", "sun-blind",
                     "stubborn", "spent", "wary"),
        adverbs=("grimly", "without rest", "against the wind", "in silence", "at first light", "too late"),
        artifacts=("a frost-etched piton", "a leather-bound summit register", "a brass altimeter",
                    "a torn expedition flag", "an ice-cracked thermos", "a hand-forged crampon"),
    ),
    "clockwork": Theme(
        name="clockwork",
        subjects=("the clockmaker", "her apprentice", "the guild inspector", "a wary customer",
                   "the tinkerer", "the foundry master", "the night watchman", "an itinerant mechanic"),
        verbs=("assembled", "dismantled", "wound", "recalibrated", "smuggled", "patented",
               "sabotaged", "restored", "distrusted", "auctioned", "reverse-engineered", "concealed"),
        objects=("the escapement", "a brass gear train", "the pressure valve", "the winding key",
                  "a coil of copper wire", "the drive spring", "the pendulum weight", "a punch-card cylinder"),
        places=("the workshop loft", "the guild archive", "the boiler room", "the clock tower stairwell",
                 "the scrapyard annex", "the patent office", "the foundry floor", "the market arcade"),
        adjectives=("precise", "soot-streaked", "meticulous", "half-finished", "ingenious",
                     "obsolete", "over-engineered", "unlicensed"),
        adverbs=("precisely", "with great care", "in secret", "against regulation", "at midnight", "reluctantly"),
        artifacts=("a brass astrolabe engine", "a punch-card ledger", "an iron thimble gauge",
                    "a phosphor lantern core", "a silver-tern automaton", "a twelve-string tension key"),
    ),
    "caravan": Theme(
        name="caravan",
        subjects=("the caravan master", "the youngest trader", "the camel driver", "a desert scout",
                   "the water-keeper", "the map-seller", "the veiled envoy", "an old guide"),
        verbs=("bartered", "rationed", "abandoned", "tracked", "distrusted", "guided", "smuggled",
               "recorded", "buried", "unearthed", "warned", "outpaced"),
        objects=("the water skins", "a bundle of dyed cloth", "the trade ledger", "a sealed manifest",
                  "the salt blocks", "a coil of tent rope", "the spice chest", "a folded star-chart"),
        places=("the dune ridge", "the oasis palms", "the caravanserai gate", "the salt flat",
                 "the hidden wadi", "the market square", "the sandstorm's edge", "the well-head"),
        adjectives=("sun-scoured", "wary", "unhurried", "dust-caked", "shrewd",
                     "weather-hardened", "tight-lipped", "patient"),
        adverbs=("without haste", "under a hard sun", "at the well's edge", "before dawn", "grudgingly", "at last"),
        artifacts=("a salt-cured ledger", "a copper astrolabe", "a wax-sealed manifest",
                    "a dune-worn compass", "a bronze water-gauge", "an inked star-chart"),
    ),
}

_INTERJECTIONS = ("Wait", "Listen", "Enough", "No matter", "So be it", "Then again")

_SENTENCE_TEMPLATES = (
    "{Subject} {verb} {object} near {place}.",
    "{Subject}, {adjective} and silent, {verb} {object}.",
    "By {place}, {subject} {verb} {object} {adverb}.",
    '"{interjection}," {subject} said, {adverb} eyeing {object}.',
    "{Subject} had long {verb} {object}, though {place} offered little comfort.",
    "It was {subject} who first {verb} {object}, {adverb}, while the others watched.",
)

_NEEDLE_TEMPLATES = (
    "{Subject} discovered {artifact} hidden beneath {place}, its surface bearing the mark {marker}.",
    "No one had touched {artifact} since {subject} first found it near {place}, faintly etched with {marker}.",
    "{Subject} recalled the day {artifact} appeared at {place}, stamped with the sign {marker}.",
    "Half-buried at {place}, {artifact} still carried the faded inscription {marker}, or so {subject} insisted.",
)


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _sentence(rng: random.Random, theme: Theme) -> str:
    template = rng.choice(_SENTENCE_TEMPLATES)
    subject = rng.choice(theme.subjects)
    return template.format(
        Subject=_cap(subject),
        subject=subject,
        verb=rng.choice(theme.verbs),
        object=rng.choice(theme.objects),
        place=rng.choice(theme.places),
        adjective=rng.choice(theme.adjectives),
        adverb=rng.choice(theme.adverbs),
        interjection=rng.choice(_INTERJECTIONS),
    )


def generate_paragraph(rng: random.Random, theme: Theme, *, n_sentences: int) -> str:
    return " ".join(_sentence(rng, theme) for _ in range(max(1, n_sentences)))


def generate_body(rng: random.Random, theme: Theme, *, target_words: int) -> list[str]:
    """Generate paragraphs until roughly `target_words` words are reached.
    Word count is measured, not estimated, so callers get a predictable
    (if approximate — the last paragraph can overshoot slightly) document
    size regardless of theme or seed."""
    paragraphs: list[str] = []
    word_count = 0
    while word_count < target_words:
        n_sentences = rng.randint(3, 7)
        para = generate_paragraph(rng, theme, n_sentences=n_sentences)
        paragraphs.append(para)
        word_count += len(para.split())
    return paragraphs


def make_needle(rng: random.Random, theme: Theme, *, marker: str) -> str:
    """A rare, thematically-plausible sentence containing a unique `marker`
    token — the ground-truth "needle" a retrieval test can search for with
    an exactly-known correct chunk. `marker` should be globally unique
    across the whole generated library (see synth/library.py)."""
    template = rng.choice(_NEEDLE_TEMPLATES)
    return template.format(
        Subject=_cap(rng.choice(theme.subjects)),
        subject=rng.choice(theme.subjects),
        artifact=rng.choice(theme.artifacts),
        place=rng.choice(theme.places),
        marker=marker,
    )


def make_mention(rng: random.Random, theme: Theme, *, name: str) -> str:
    """A sentence naming `name` (a character's surface form for this
    chapter — see doubles.EntityMention) doing something thematic, so the
    name appears as a literal, findable token/phrase in real chunk text."""
    template = rng.choice(
        (
            "{name} {verb} {object} {adverb}.",
            "{name} stood near {place}, {adjective} and watchful.",
            "It was {name} who {verb} {object} first, near {place}.",
        )
    )
    return template.format(
        name=name,
        verb=rng.choice(theme.verbs),
        object=rng.choice(theme.objects),
        place=rng.choice(theme.places),
        adjective=rng.choice(theme.adjectives),
        adverb=rng.choice(theme.adverbs),
    )


def wrap_lines(text: str, width: int = 90) -> list[str]:
    """Word-wrap for backends (PDF) that need pre-wrapped lines rather than
    a text box with automatic reflow."""
    lines: list[str] = []
    for para in text.split("\n"):
        wrapped = textwrap.wrap(para, width=width) or [""]
        lines.extend(wrapped)
    return lines
