"""Deliberately duplicates server/app/rules_provider.py's shape rather
than importing it — same reasoning as mcp_server/corpus_provider.py's
own docstring: this MCP server and the HTTP server are two
independently deployable artifacts over the same library.

The domain pack registry is empty by default, same principle as the
HTTP server's own provider: this MCP server has zero hard dependency
on any specific domain pack. A real deployment wanting Lal Kitab rules
served would pass its own `domain_pack_registry_provider` into
`build_server()`.
"""

from __future__ import annotations

import os
from functools import lru_cache

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
            "this MCP server should open for rule storage (e.g. ./data/rules.sqlite3)"
        )
    return SqliteRuleStore(db_path)


def get_rule_store() -> SqliteRuleStore:
    return _rule_store_singleton()


def get_domain_pack_registry() -> dict[str, DomainPack]:
    return {}


def get_rule_extraction_pipeline() -> tuple[RuleCandidateDetector, RuleExtractor, RuleVerifier]:
    raise RuntimeError(
        "no rule extraction pipeline configured for this MCP server — pass "
        "rule_extraction_pipeline_provider to build_server()"
    )
