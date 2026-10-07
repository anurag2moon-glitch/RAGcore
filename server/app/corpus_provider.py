"""get_corpus — a FastAPI dependency wrapping a single, process-wide
Corpus instance, built once from environment variables.

Single-tenant, by design — see docs/adr/0008-api-server-as-sibling-package.md
for why multi-tenant HTTP routing isn't attempted in this first pass. A
test overrides this entirely via `app.dependency_overrides[get_corpus]`,
never by setting these env vars, so tests never need real Voyage/LanceDB
credentials.
"""

from __future__ import annotations

import os
from functools import lru_cache

from ragcore.corpus import Corpus

RAGCORE_DB_PATH_ENV = "RAGCORE_DB_PATH"


@lru_cache(maxsize=1)
def _corpus_singleton() -> Corpus:
    db_path = os.environ.get(RAGCORE_DB_PATH_ENV)
    if not db_path:
        raise RuntimeError(
            f"{RAGCORE_DB_PATH_ENV} must be set to the on-disk LanceDB path "
            "this server should open (e.g. ./data/corpus.lancedb)"
        )
    return Corpus.open(db_path)


def get_corpus() -> Corpus:
    return _corpus_singleton()
