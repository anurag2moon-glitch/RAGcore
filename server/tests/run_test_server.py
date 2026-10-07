"""Runs the real server/ FastAPI app, with its Corpus dependency
overridden to the same in-memory test doubles test_api.py uses, as a
real uvicorn process listening on a real TCP socket.

For sibling-package test suites that can't use an in-process ASGI
bridge the way server/tests/test_api.py does (TestClient) — e.g.
sdk/typescript/'s, which needs to exercise this SDK's actual HTTP
requests, not Python internals. Spawned as a subprocess, never
imported: `python server/tests/run_test_server.py <port>`. Prints
"READY" to stdout once bound, so a caller polling for that line knows
when to start sending requests rather than guessing a fixed startup
delay.
"""

from __future__ import annotations

import sys

import uvicorn
from app.corpus_provider import get_corpus
from app.main import app
from app.rules_provider import get_rule_store
from fakes import FakeEmbedder, FakeGenerator, InMemoryStore

from ragcore.corpus import Corpus
from ragcore.rules.sqlite_store import SqliteRuleStore
from ragcore.rules.types import Effect, FactPredicate, Rule


class _ReadyLoggingServer(uvicorn.Server):
    async def startup(self, sockets: list | None = None) -> None:
        await super().startup(sockets=sockets)
        print("READY", flush=True)


def _sample_rule(rule_id: str, *, review_status: str = "verified") -> Rule:
    return Rule(
        rule_id=rule_id, domain="test", pack_id="p", pack_version="1", book_id="b1",
        page_start=1, chunk_ids=("c1",), verbatim_span="Mars in the 8th house causes trouble.",
        language="en", normalized_statement="n",
        conditions=FactPredicate(fact="planet.mars.house", operator="eq", value=8),
        effect=Effect(category="health", polarity="negative"), rule_type="HARD",
        confidence=0.9, extraction_model="m", extraction_version="v1", created_at="now",
        review_status=review_status,
    )


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    with_generator = "--with-generator" in sys.argv
    with_sample_rules = "--with-sample-rules" in sys.argv

    corpus = Corpus(
        InMemoryStore(), FakeEmbedder(), generator=FakeGenerator() if with_generator else None
    )
    app.dependency_overrides[get_corpus] = lambda: corpus

    rule_store = SqliteRuleStore(":memory:")
    if with_sample_rules:
        rule_store.add_rules([
            _sample_rule("rule:sample_verified"),
            _sample_rule("rule:sample_pending", review_status="auto"),
        ])
    app.dependency_overrides[get_rule_store] = lambda: rule_store

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    _ReadyLoggingServer(config).run()


if __name__ == "__main__":
    main()
