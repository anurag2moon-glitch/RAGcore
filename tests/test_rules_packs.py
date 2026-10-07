"""export_pack / to_json / from_json / import_pack: pack-version
scoping, JSON round-tripping, idempotent re-import, and the tenant-
mismatch guard against a cross-tenant rule_id collision — Phase 10 §5.
"""

from __future__ import annotations

import pytest

from ragcore.rules.packs import (
    PackImportError,
    export_pack,
    from_json,
    import_pack,
    to_json,
)
from ragcore.rules.sqlite_store import SqliteRuleStore
from ragcore.rules.types import Effect, FactPredicate, Rule


def _rule(
    rule_id: str, *, pack_id: str = "pack1", pack_version: str = "1.0",
    tenant_scope: str = "global", tenant_id: str | None = None, book_id: str = "b1",
) -> Rule:
    return Rule(
        rule_id=rule_id, domain="test", pack_id=pack_id, pack_version=pack_version, book_id=book_id,
        page_start=1, chunk_ids=("c:0",), verbatim_span=f"span {rule_id}", language="en",
        normalized_statement="n", conditions=FactPredicate(fact="x", operator="eq", value=8),
        effect=Effect(category="health", polarity="negative"), rule_type="HARD",
        confidence=0.9, extraction_model="m", extraction_version="v1", created_at="now",
        tenant_scope=tenant_scope, tenant_id=tenant_id,
    )


# -- export_pack scoping ----------------------------------------------------


def test_export_pack_only_includes_the_requested_pack_version() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([
        _rule("r1", pack_version="1.0"),
        _rule("r2", pack_version="2.0", book_id="b2"),
    ])

    bundle = export_pack(store, domain="test", pack_id="pack1", pack_version="1.0")

    assert [r.rule_id for r in bundle.rules] == ["r1"]


def test_export_pack_only_includes_the_requested_pack_id() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([
        _rule("r1", pack_id="pack1"),
        _rule("r2", pack_id="pack2", book_id="b2"),
    ])

    bundle = export_pack(store, domain="test", pack_id="pack1", pack_version="1.0")

    assert [r.rule_id for r in bundle.rules] == ["r1"]


def test_export_pack_scopes_by_tenant_too() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([
        _rule("global_rule", tenant_scope="global"),
        _rule("tenant_a_rule", tenant_scope="tenant", tenant_id="tenantA", book_id="b2"),
    ])

    global_bundle = export_pack(store, domain="test", pack_id="pack1", pack_version="1.0")
    tenant_bundle = export_pack(
        store, domain="test", pack_id="pack1", pack_version="1.0",
        tenant_scope="tenant", tenant_id="tenantA",
    )

    assert [r.rule_id for r in global_bundle.rules] == ["global_rule"]
    assert [r.rule_id for r in tenant_bundle.rules] == ["tenant_a_rule"]


# -- JSON round-trip ----------------------------------------------------


def test_to_json_then_from_json_round_trips_every_field() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1")])
    bundle = export_pack(store, domain="test", pack_id="pack1", pack_version="1.0")

    restored = from_json(to_json(bundle))

    assert restored.pack_id == bundle.pack_id
    assert restored.pack_version == bundle.pack_version
    assert restored.domain == bundle.domain
    assert restored.tenant_scope == bundle.tenant_scope
    assert restored.rules == bundle.rules


def test_from_json_rejects_an_unknown_schema_version() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1")])
    bundle = export_pack(store, domain="test", pack_id="pack1", pack_version="1.0")
    payload = to_json(bundle).replace('"schema_version": "1"', '"schema_version": "99"')

    with pytest.raises(PackImportError, match="schema_version"):
        from_json(payload)


# -- import_pack ----------------------------------------------------------


def test_import_pack_writes_every_rule_and_returns_the_count() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1"), _rule("r2", book_id="b2")])
    bundle = export_pack(store, domain="test", pack_id="pack1", pack_version="1.0")

    target = SqliteRuleStore(":memory:")
    count = import_pack(target, bundle)

    assert count == 2
    assert target.get("r1") is not None
    assert target.get("r2") is not None


def test_importing_the_same_bundle_twice_is_idempotent() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1")])
    bundle = export_pack(store, domain="test", pack_id="pack1", pack_version="1.0")

    target = SqliteRuleStore(":memory:")
    import_pack(target, bundle)
    import_pack(target, bundle)

    assert len(target.rules_for_domain("test")) == 1


def test_a_global_bundle_imports_under_any_tenant_id() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1", tenant_scope="global")])
    bundle = export_pack(store, domain="test", pack_id="pack1", pack_version="1.0")

    target = SqliteRuleStore(":memory:")
    count = import_pack(target, bundle, tenant_id="whoever")

    assert count == 1


def test_a_tenant_bundle_imports_under_its_own_matching_tenant_id() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1", tenant_scope="tenant", tenant_id="tenantA")])
    bundle = export_pack(
        store, domain="test", pack_id="pack1", pack_version="1.0",
        tenant_scope="tenant", tenant_id="tenantA",
    )

    target = SqliteRuleStore(":memory:")
    count = import_pack(target, bundle, tenant_id="tenantA")

    assert count == 1


def test_a_tenant_bundle_refuses_import_under_a_different_tenant_id() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1", tenant_scope="tenant", tenant_id="tenantA")])
    bundle = export_pack(
        store, domain="test", pack_id="pack1", pack_version="1.0",
        tenant_scope="tenant", tenant_id="tenantA",
    )

    target = SqliteRuleStore(":memory:")
    with pytest.raises(PackImportError, match="tenant"):
        import_pack(target, bundle, tenant_id="tenantB")

    assert target.get("r1") is None  # refused before any write
