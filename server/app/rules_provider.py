"""FastAPI dependencies for Phase 10's rule layer: a process-wide
RuleStore (mirrors app.corpus_provider's Corpus singleton exactly —
same env-var-keyed lazy construction, same test-override contract) and
a **pluggable domain pack registry**.

The registry is deliberately empty by default: this server has zero
hard dependency on any specific domain pack (Lal Kitab or otherwise) —
the same domain-pluggable principle docs/adr/0013 established for
ragcore.rules itself, extended to the HTTP layer. A real deployment
wanting to serve Lal Kitab rules registers it explicitly (e.g. in a
startup hook: `app.dependency_overrides[get_domain_pack_registry] =
lambda: {"lal_kitab": LalKitabPack()}`), the identical override
mechanism tests already use for get_corpus — this server's own
pyproject.toml deliberately does not depend on domains/lal_kitab.
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import Depends, HTTPException

from ragcore.rules.base import RuleCandidateDetector, RuleExtractor, RuleVerifier
from ragcore.rules.domain import DomainPack
from ragcore.rules.sqlite_store import SqliteRuleStore

RAGCORE_RULES_DB_PATH_ENV = "RAGCORE_RULES_DB_PATH"


@lru_cache(maxsize=1)
def _rule_store_singleton() -> SqliteRuleStore:
    db_path = os.environ.get(RAGCORE_RULES_DB_PATH_ENV)
    if not db_path:
        raise RuntimeError(
            f"{RAGCORE_RULES_DB_PATH_ENV} must be set to the on-disk SQLite path "
            "this server should open for rule storage (e.g. ./data/rules.sqlite3)"
        )
    return SqliteRuleStore(db_path)


def get_rule_store() -> SqliteRuleStore:
    return _rule_store_singleton()


def get_domain_pack_registry() -> dict[str, DomainPack]:
    return {}


def get_domain_pack(
    domain: str,
    registry: dict[str, DomainPack] = Depends(get_domain_pack_registry),  # noqa: B008
) -> DomainPack:
    pack = registry.get(domain)
    if pack is None:
        raise HTTPException(
            status_code=404,
            detail=f"no domain pack registered for domain {domain!r} "
            f"(registered: {sorted(registry)})",
        )
    return pack


def get_rule_extraction_pipeline() -> tuple[RuleCandidateDetector, RuleExtractor, RuleVerifier]:
    """No default implementation — unlike get_domain_pack_registry
    (an empty dict is a perfectly valid, inert default), a missing
    extraction pipeline can't be answered with "nothing registered,
    404" because /rules/extract has no meaningful partial behavior
    without one. A real deployment overrides this with
    ClaudeRuleCandidateDetector/ClaudeRuleExtractor/ClaudeRuleVerifier
    (real Claude API calls, so never constructed eagerly at import
    time); tests override it with fakes, the same
    app.dependency_overrides mechanism every other provider here
    uses."""
    raise HTTPException(
        status_code=503,
        detail="no rule extraction pipeline configured on this server",
    )
