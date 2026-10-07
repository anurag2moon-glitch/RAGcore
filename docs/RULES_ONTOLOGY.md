# Rule ontology reference

Phase 10 §8 deliverable. How `ragcore.rules` stays domain-agnostic —
the `DomainPack`/`OntologyNormalizer`/`FactTypeSpec` contract a
concrete domain implements — plus the first (and so far only) concrete
pack, `domains/lal_kitab/`, as the worked reference. See
`docs/adr/0013-rule-extraction-domain-pluggable-architecture.md` for
the architecture decision this implements.

## The contract (`ragcore/rules/domain.py`)

A domain pack is **inert data plus pure functions** — never itself an
extraction pipeline. `ragcore.rules.pipeline.build_rules` drives
extraction; a pack only supplies domain knowledge.

### `FactTypeSpec`

One entry in a domain's fact-type registry — what a given
`Condition.fact` key means and what values are legal for it.

```python
FactTypeSpec(
    fact_key="planet.mars.house",
    value_type="int",   # "int" | "float" | "str" | "bool" | "enum"
    enum_values=(),      # only for value_type="enum" - the closed set of legal values
    description="The house (1-12) Mars occupies.",
)
```

`validate_value(value)` is used by `ragcore.rules.evaluator.validate_facts`
to catch a malformed fact set before matching — e.g. a `bool` is
explicitly rejected for an `int`-typed fact, since `bool` is a Python
subclass of `int` and would otherwise silently pass `isinstance`.

### `OntologyNormalizer`

Two methods, both allowed to return `None` — "unrecognized," never a
guess:

```python
class OntologyNormalizer(Protocol):
    def normalize_fact_key(self, raw_mention: str) -> str | None:
        """Raw source-text mention -> canonical fact key, e.g.
        "mangal ka ghar" -> "planet.mars.house"."""

    def normalize_fact_value(self, fact_key: str, raw_mention: str) -> Any | None:
        """Raw mention -> canonical value for that fact key, e.g.
        normalize_fact_value("planet.mars.house", "ashtam bhava") -> 8."""
```

This is the seam `ragcore.rules.claude.ClaudeRuleExtractor` calls
after the model proposes a `(fact_mention, operator, value_mention)`
triple in the source's own vocabulary (Phase 10 §2's extraction
prompt deliberately tells the model *not* to normalize anything
itself) — see `_build_conditions` in that module. If either call
returns `None`, the whole candidate rule is dropped rather than stored
with a partial/wrong condition — "no span, no rule" applied to
conditions, not just the `verbatim_span` field.

### `DomainPack`

```python
class DomainPack(Protocol):
    domain: str          # stable slug, becomes Rule.domain
    pack_id: str
    pack_version: str
    ontology: OntologyNormalizer

    def fact_types(self) -> tuple[FactTypeSpec, ...]: ...
    def extraction_system_prompt(self) -> str: ...     # passes 1-2
    def verification_system_prompt(self) -> str: ...   # pass 3
```

## Worked example: `domains/lal_kitab/`

### Vocabulary (`lal_kitab.vocabulary`)

A hand-authored, closed table — 9 planets (Navagraha), 12 houses (by
their Hindi/Sanskrit ordinal names, e.g. `"ashtam bhava"` for the 8th
— the exact example `OntologyNormalizer.normalize_fact_value`'s own
docstring uses), 12 rashis (signs), 4 dignity states. Every alias/
transliteration this pack currently recognizes lives in this one file
— extend it as real source texts surface spellings not covered yet,
rather than guessing ahead of real material.

### Fact key shapes `LalKitabOntologyNormalizer` produces

| Shape | Example | Meaning |
|---|---|---|
| `planet.{p}.house` | `planet.mars.house` | int 1-12 |
| `planet.{p}.sign` | `planet.mars.sign` | enum, one of the 12 signs |
| `planet.{p}.retrograde` | `planet.saturn.retrograde` | bool |
| `planet.{p}.dignity` | `planet.mars.dignity` | enum: exalted/debilitated/own/neutral |
| `planet.{p}.blind` / `.sleeping` / `.pakka_ghar` | `planet.saturn.blind` | bool — Phase 10's own "blind/sleeping/pakka-ghar status" |
| `planet.{a}.conjunct.{b}` | `planet.mars.conjunct.saturn` | bool, **symmetric** — canonicalized `a < b` alphabetically regardless of mention order |
| `planet.{a}.aspects.{b}` | `planet.mars.aspects.saturn` | bool, **directional** — `a` is whichever planet the mention names first |
| `house.{1-12}.lord` | `house.7.lord` | enum, one of the 9 planets — note the *house number* is the key's subject, the *planet* is the value |
| `native.dasha_lord` | — | enum, one of the 9 planets — checked **before** any planet-specific branch, since a dasha mention also names a planet ("mangal dasha") but means something else entirely |
| `native.age` | — | int |

`pack.fact_types()` generates all 185 of these from the vocabulary
(7 per planet + `C(9,2)=36` conjunction pairs + `9×8=72` directed
aspect pairs + 12 house-lordship facts + dasha/age) rather than
hand-enumerating them — this is what keeps the registry in sync with
`vocabulary.PLANETS` automatically if a tenth graha were ever added.

### Resolution order, and why it matters

`normalize_fact_key` checks, in this order: dasha/age mentions first
(must win over the planet-specific branch below even though they also
name a planet) → lordship mentions (subject is a house number, found
via `HOUSE_ORDINAL_ALIASES`, not a planet) → conjunction/aspect
(two-planet mentions, via `_find_planets_in_order`) → single-planet
fact types, themselves checked specific-before-general
(`pakka_ghar` before the generic `house` check, since the phrase
"pakka ghar" itself contains the substring "ghar").

Both ordering decisions were real bugs caught by direct execution
before being codified as this order, not designed correctly on the
first attempt — see `lal_kitab/ontology.py`'s own module docstring and
`tests/test_ontology.py`'s `test_dasha_mention_resolves_to_native_dasha_lord_not_a_planet_fact`
regression test for the dasha case specifically.

### Extraction/verification prompts

`extraction_system_prompt()` tells the model to use the source's own
mixed Hindi/English vocabulary verbatim and *not* normalize anything
itself (normalization is this pack's job, applied after extraction).
`verification_system_prompt()` tells the independent verifier that Lal
Kitab's own style states predictions in absolute language even when
the actual condition is narrower than it first appears — check that
every stated condition, not just the effect, is actually present in
the verbatim span.

## Adding a second domain pack

Nothing in `ragcore.rules` core needs to change — implement
`DomainPack`/`OntologyNormalizer` as a new sibling package (same
`domains/<name>/` shape as `domains/lal_kitab/`: its own
`pyproject.toml` depending on `ragcore[generation]`, its own tests,
its own CI job in `.github/workflows/ci.yml`), register it with
whichever server/MCP tool's domain pack registry you're running
(`app.dependency_overrides[get_domain_pack_registry]` for the HTTP
server, the `domain_pack_registry_provider` constructor argument for
the MCP server), and the entire extraction/consistency/review/serving
pipeline works unchanged — this is the concrete claim
`domains/lal_kitab/tests/test_integration.py` already proves for Lal
Kitab specifically.
