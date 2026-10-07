"""get_corpus — a FastAPI dependency wrapping a single, process-wide
Corpus instance, built once from environment variables. get_scoped_corpus
layers tenant scoping on top of it via app.auth.get_tenant_id — see that
module's own docstring for the opt-in, backward-compatible design.

A test overrides get_corpus entirely via
`app.dependency_overrides[get_corpus]`, never by setting these env
vars, so tests never need real Voyage/LanceDB credentials.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from app.auth import get_tenant_id
from app.settings import get_settings
from ragcore.corpus import Corpus
from ragcore.tenancy import TenantScopedCorpus


@lru_cache(maxsize=1)
def _corpus_singleton() -> Corpus:
    db_path = get_settings().ragcore_db_path
    if not db_path:
        raise RuntimeError(
            "RAGCORE_DB_PATH must be set to the on-disk LanceDB path "
            "this server should open (e.g. ./data/corpus.lancedb)"
        )
    return Corpus.open(db_path)


def get_corpus() -> Corpus:
    return _corpus_singleton()


def get_scoped_corpus(
    corpus: Corpus = Depends(get_corpus),  # noqa: B008 - FastAPI's own DI idiom
    tenant_id: str | None = Depends(get_tenant_id),
) -> Corpus | TenantScopedCorpus:
    """`tenant_id` is `None` whenever RAGCORE_API_KEYS is unset (auth
    disabled) — every route gets the raw, unscoped `Corpus`, identical
    to this server's behavior before auth existed. Once a tenant_id is
    resolved from a valid API key, every route gets a fresh
    `TenantScopedCorpus` instead, which forces that tenant_id on every
    call it makes (ragcore.tenancy's own enforcement, not duplicated
    here)."""
    if tenant_id is None:
        return corpus
    return TenantScopedCorpus(corpus, tenant_id)
