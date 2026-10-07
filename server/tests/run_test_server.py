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
from fakes import FakeEmbedder, FakeGenerator, InMemoryStore

from ragcore.corpus import Corpus


class _ReadyLoggingServer(uvicorn.Server):
    async def startup(self, sockets: list | None = None) -> None:
        await super().startup(sockets=sockets)
        print("READY", flush=True)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    with_generator = "--with-generator" in sys.argv

    corpus = Corpus(
        InMemoryStore(), FakeEmbedder(), generator=FakeGenerator() if with_generator else None
    )
    app.dependency_overrides[get_corpus] = lambda: corpus

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    _ReadyLoggingServer(config).run()


if __name__ == "__main__":
    main()
