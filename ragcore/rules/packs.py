"""Rule packs: export/import as versioned bundles (Phase 10 §5 -
"Rule packs: export/import as versioned bundles (global packs shared
across tenants, private packs per tenant)").

**Tenant-scope limitation, stated plainly**: a `Rule`'s `rule_id`
(`rule_content_hash`) deliberately does **not** include `tenant_id` —
see `ragcore.rules.types`'s own docstring. That's correct for a global
pack (the whole point is one shared id across every tenant using it),
but it means a `tenant`-scoped bundle's rule_ids are only ever safe to
import back into the **same** `tenant_id` it was exported for:
importing it into a different tenant would try to insert the same
`rule_id` under a different owner, and `RuleStore.add_rules`'s
insert-or-ignore semantics would silently keep whichever tenant's copy
landed first rather than raising. `import_pack` refuses that case
outright (`PackImportError`) instead of risking it — true cross-tenant
*copying* of a private pack (not sharing one global id, but minting
each tenant its own distinct rule_ids for the "same" imported content)
isn't implemented, since nothing in the spec requires it, and doing it
safely needs a real tenant-aware id-rewrite scheme this layer doesn't
have a settled design for yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from ragcore.errors import RagcoreError
from ragcore.rules.types import Rule

if TYPE_CHECKING:
    from ragcore.rules.base import RuleStore

PACK_BUNDLE_SCHEMA_VERSION = "1"


class PackImportError(RagcoreError):
    pass


@dataclass(frozen=True)
class RulePackBundle:
    pack_id: str
    pack_version: str
    domain: str
    tenant_scope: Literal["global", "tenant"]
    tenant_id: str | None
    rules: tuple[Rule, ...]
    exported_at: str
    schema_version: str = PACK_BUNDLE_SCHEMA_VERSION


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def export_pack(
    store: RuleStore,
    *,
    domain: str,
    pack_id: str,
    pack_version: str,
    tenant_scope: Literal["global", "tenant"] = "global",
    tenant_id: str | None = None,
    latest_only: bool = True,
) -> RulePackBundle:
    """Every rule in `store` matching (domain, pack_id, pack_version,
    tenant_scope, tenant_id) exactly — a bundle is always scoped to
    one pack version, never "the domain's current state", so an
    exported bundle's contents are reproducible regardless of what's
    extracted into the store afterward."""
    rules = tuple(
        r for r in store.rules_for_domain(domain, latest_only=latest_only)
        if r.pack_id == pack_id and r.pack_version == pack_version
        and r.tenant_scope == tenant_scope and r.tenant_id == tenant_id
    )
    return RulePackBundle(
        pack_id=pack_id, pack_version=pack_version, domain=domain,
        tenant_scope=tenant_scope, tenant_id=tenant_id, rules=rules, exported_at=_now(),
    )


def to_json(bundle: RulePackBundle) -> str:
    return json.dumps(
        {
            "schema_version": bundle.schema_version,
            "pack_id": bundle.pack_id,
            "pack_version": bundle.pack_version,
            "domain": bundle.domain,
            "tenant_scope": bundle.tenant_scope,
            "tenant_id": bundle.tenant_id,
            "exported_at": bundle.exported_at,
            "rules": [json.loads(r.model_dump_json()) for r in bundle.rules],
        }
    )


def from_json(payload: str) -> RulePackBundle:
    data = json.loads(payload)
    if data.get("schema_version") != PACK_BUNDLE_SCHEMA_VERSION:
        raise PackImportError(
            f"unsupported pack bundle schema_version {data.get('schema_version')!r} "
            f"(expected {PACK_BUNDLE_SCHEMA_VERSION!r})"
        )
    return RulePackBundle(
        pack_id=data["pack_id"], pack_version=data["pack_version"], domain=data["domain"],
        tenant_scope=data["tenant_scope"], tenant_id=data["tenant_id"],
        exported_at=data["exported_at"], schema_version=data["schema_version"],
        rules=tuple(Rule.model_validate(r) for r in data["rules"]),
    )


def import_pack(store: RuleStore, bundle: RulePackBundle, *, tenant_id: str | None = None) -> int:
    """Writes every rule in `bundle` into `store` via `add_rules`
    (idempotent by content-addressed rule_id — importing the same
    bundle twice is a safe no-op, not a duplicate). Returns the number
    of rules written (including no-op re-imports, since add_rules
    doesn't distinguish — this is a count of what the bundle
    *contains*, not a count of new rows).

    `tenant_id` is the importing context, required to match a
    `tenant`-scoped bundle's own `tenant_id` exactly — see this
    module's docstring for why a mismatch is refused rather than
    silently risking a cross-tenant rule_id collision. Irrelevant for
    a `global` bundle, which may be imported into any context (that's
    the point of a global pack)."""
    if bundle.tenant_scope == "tenant" and tenant_id != bundle.tenant_id:
        raise PackImportError(
            f"bundle is tenant-scoped to {bundle.tenant_id!r}, cannot import under "
            f"tenant_id={tenant_id!r} - cross-tenant import of a private pack isn't "
            f"supported (see ragcore.rules.packs module docstring)"
        )
    store.add_rules(list(bundle.rules))
    return len(bundle.rules)
