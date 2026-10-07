"""DomainPack protocol (Phase 10 §7): the interface a concrete domain
(Lal Kitab, BPHS, ...) implements to plug into the generic rule
pipeline without the pipeline ever needing domain-specific code — see
docs/adr/0013-rule-extraction-domain-pluggable-architecture.md.

A DomainPack is inert data plus pure functions — ontology mappings, a
fact-type registry, prompt templates — never itself an extraction
pipeline. `ragcore.rules.pipeline` drives extraction, calling into a
pack only for domain knowledge, which is what makes "add a new domain
without touching core code" actually true rather than aspirational.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

FactValueType = Literal["int", "float", "str", "bool", "enum"]


@dataclass(frozen=True)
class FactTypeSpec:
    """One entry in a domain's fact-type registry — Phase 10 §7's
    "evaluator fact types." Tells the evaluator (`ragcore.rules.evaluator`)
    and, optionally, an extraction prompt what a given `Condition.fact`
    key means and what values are valid for it."""

    fact_key: str
    value_type: FactValueType
    enum_values: tuple[str, ...] = ()
    """Only meaningful when `value_type == "enum"` — the closed set of
    valid values (e.g. a sign name)."""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.fact_key.strip():
            raise ValueError("fact_key must be a non-empty string")
        if self.value_type == "enum" and not self.enum_values:
            raise ValueError("value_type='enum' requires at least one enum_values entry")
        if self.value_type != "enum" and self.enum_values:
            raise ValueError(f"enum_values is only valid when value_type='enum', got {self.value_type!r}")

    def validate_value(self, value: Any) -> bool:
        """Whether `value` is a legal value for this fact. Used by
        `ragcore.rules.evaluator.validate_facts` to catch a malformed
        fact set before it's matched against any rule, and usable by a
        domain pack's own extraction-time sanity checks too."""
        if self.value_type == "enum":
            return value in self.enum_values
        checkers: dict[FactValueType, type | tuple[type, ...]] = {
            "int": int,
            "float": (int, float),
            "str": str,
            "bool": bool,
        }
        expected = checkers[self.value_type]
        # bool is a subclass of int in Python — exclude it explicitly so an
        # int-typed fact doesn't silently accept True/False.
        if self.value_type == "int" and isinstance(value, bool):
            return False
        return isinstance(value, expected)


@runtime_checkable
class OntologyNormalizer(Protocol):
    """Maps raw source-text mentions (aliases, transliterations —
    Phase 10 §2's "Normalize vocabulary through a domain ontology:
    aliases, transliterations... mapped to canonical planet, house and
    sign IDs") to the canonical fact keys/values a domain's evaluator
    understands."""

    def normalize_fact_key(self, raw_mention: str) -> str | None:
        """The canonical `Condition.fact` key for a raw mention (e.g.
        "mangal" -> "planet.mars.house"), or `None` if unrecognized —
        never a guess."""
        ...

    def normalize_fact_value(self, fact_key: str, raw_mention: str) -> Any | None:
        """The canonical value for a raw mention, given the fact key
        it's a value for (e.g. normalizing "ashtam bhava" to the int
        `8` for a `*.house` fact), or `None` if unrecognized."""
        ...


@runtime_checkable
class DomainPack(Protocol):
    domain: str
    """Short, stable slug (e.g. "lal_kitab") — becomes `Rule.domain`."""

    pack_id: str
    pack_version: str
    ontology: OntologyNormalizer

    def fact_types(self) -> tuple[FactTypeSpec, ...]:
        """The domain's complete fact-type registry — every
        `Condition.fact` key a rule from this pack may legally use."""
        ...

    def extraction_system_prompt(self) -> str:
        """The domain-specific system prompt guiding candidate-rule
        detection and structured extraction (Phase 10 §2, passes 1-2).
        The generic pipeline supplies the chunk text and the output
        schema; this supplies the domain's own vocabulary, examples,
        and extraction conventions."""
        ...

    def verification_system_prompt(self) -> str:
        """The domain-specific system prompt for the independent
        verification pass (Phase 10 §2, pass 3) — checking that a
        candidate rule's `verbatim_span` actually supports its
        `conditions` and `effect`."""
        ...
