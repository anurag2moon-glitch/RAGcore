"""ragcore.rules.domain: FactTypeSpec's validation (both directions —
a valid value accepted, every invalid shape rejected) and that a
plain, non-inheriting class genuinely satisfies the DomainPack/
OntologyNormalizer Protocols at runtime, not just by type-checker
convention — Phase 10 §7.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from ragcore.rules.domain import DomainPack, FactTypeSpec, OntologyNormalizer


class _FakeOntology:
    _KEY_ALIASES: ClassVar[dict[str, str]] = {"mangal": "planet.mars.house"}
    _VALUE_ALIASES: ClassVar[dict[tuple[str, str], int]] = {("planet.mars.house", "ashtam bhava"): 8}

    def normalize_fact_key(self, raw_mention: str) -> str | None:
        return self._KEY_ALIASES.get(raw_mention)

    def normalize_fact_value(self, fact_key: str, raw_mention: str):
        return self._VALUE_ALIASES.get((fact_key, raw_mention))


class _FakeDomainPack:
    domain = "test_domain"
    pack_id = "test_pack"
    pack_version = "1.0.0"
    ontology = _FakeOntology()

    def fact_types(self) -> tuple[FactTypeSpec, ...]:
        return (
            FactTypeSpec(fact_key="planet.mars.house", value_type="int"),
            FactTypeSpec(fact_key="planet.mars.sign", value_type="enum", enum_values=("aries", "taurus")),
        )

    def extraction_system_prompt(self) -> str:
        return "extract rules"

    def verification_system_prompt(self) -> str:
        return "verify rules"


# -- FactTypeSpec construction validation ------------------------------


def test_fact_type_spec_rejects_an_empty_fact_key() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FactTypeSpec(fact_key="  ", value_type="int")


def test_fact_type_spec_enum_requires_enum_values() -> None:
    with pytest.raises(ValueError, match="requires at least one"):
        FactTypeSpec(fact_key="x", value_type="enum")


def test_fact_type_spec_non_enum_rejects_enum_values() -> None:
    with pytest.raises(ValueError, match="only valid when"):
        FactTypeSpec(fact_key="x", value_type="int", enum_values=("a",))


# -- FactTypeSpec.validate_value -----------------------------------------


def test_validate_value_accepts_a_correct_int() -> None:
    spec = FactTypeSpec(fact_key="x", value_type="int")
    assert spec.validate_value(8) is True


def test_validate_value_rejects_a_bool_for_int_type() -> None:
    """bool is a subclass of int in Python - must not silently pass an
    int-typed fact check."""
    spec = FactTypeSpec(fact_key="x", value_type="int")
    assert spec.validate_value(True) is False


def test_validate_value_rejects_a_string_for_int_type() -> None:
    spec = FactTypeSpec(fact_key="x", value_type="int")
    assert spec.validate_value("8") is False


def test_validate_value_float_accepts_int_too() -> None:
    spec = FactTypeSpec(fact_key="x", value_type="float")
    assert spec.validate_value(8) is True
    assert spec.validate_value(8.5) is True


def test_validate_value_accepts_a_valid_enum_member() -> None:
    spec = FactTypeSpec(fact_key="x", value_type="enum", enum_values=("aries", "taurus"))
    assert spec.validate_value("aries") is True


def test_validate_value_rejects_an_unknown_enum_member() -> None:
    spec = FactTypeSpec(fact_key="x", value_type="enum", enum_values=("aries", "taurus"))
    assert spec.validate_value("gemini") is False


def test_validate_value_str_type() -> None:
    spec = FactTypeSpec(fact_key="x", value_type="str")
    assert spec.validate_value("hello") is True
    assert spec.validate_value(5) is False


def test_validate_value_bool_type() -> None:
    spec = FactTypeSpec(fact_key="x", value_type="bool")
    assert spec.validate_value(True) is True
    assert spec.validate_value(1) is False


# -- Protocol satisfaction ------------------------------------------------


def test_fake_domain_pack_satisfies_the_domain_pack_protocol() -> None:
    pack = _FakeDomainPack()
    assert isinstance(pack, DomainPack)


def test_fake_ontology_satisfies_the_ontology_normalizer_protocol() -> None:
    assert isinstance(_FakeOntology(), OntologyNormalizer)


def test_domain_pack_fact_types_round_trip() -> None:
    pack = _FakeDomainPack()
    fact_types = {ft.fact_key: ft for ft in pack.fact_types()}
    assert fact_types["planet.mars.house"].value_type == "int"
    assert fact_types["planet.mars.sign"].enum_values == ("aries", "taurus")


def test_domain_pack_ontology_normalizes_a_known_alias() -> None:
    pack = _FakeDomainPack()
    assert pack.ontology.normalize_fact_key("mangal") == "planet.mars.house"
    assert pack.ontology.normalize_fact_key("unknown_term") is None


def test_domain_pack_ontology_normalizes_a_known_value_alias() -> None:
    pack = _FakeDomainPack()
    assert pack.ontology.normalize_fact_value("planet.mars.house", "ashtam bhava") == 8
    assert pack.ontology.normalize_fact_value("planet.mars.house", "unknown") is None
