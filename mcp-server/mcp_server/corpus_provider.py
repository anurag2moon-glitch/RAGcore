"""get_corpus — a process-wide, lazily-built single Corpus, built once
from environment variables.

Deliberately duplicates server/app/corpus_provider.py's shape rather
than importing it: this MCP server and the HTTP server
(docs/adr/0008-api-server-as-sibling-package.md) are two independently
deployable artifacts over the same library, launched by different
clients (an MCP client like Claude Desktop/Code vs. an HTTP caller) —
having one import the other would couple two things meant to be
independently runnable. Single-tenant, no auth, same reasoning as ADR
0008 for why that isn't attempted here either.
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
            "this MCP server should open (e.g. ./data/corpus.lancedb)"
        )
    return Corpus.open(db_path)


def get_corpus() -> Corpus:
    return _corpus_singleton()
