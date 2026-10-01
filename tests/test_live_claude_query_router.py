"""Live smoke test against the real Claude API. Excluded by default (see
pyproject.toml `addopts = "-m 'not live'"`); run explicitly with
`pytest -m live`. Requires ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_claude_query_router_picks_a_real_available_mode() -> None:
    from ragcore.retrieval.routing import ClaudeQueryRouter

    router = ClaudeQueryRouter()
    modes = router.route(
        "Who are Captain Ahab's direct allies on the ship?",
        ["vector", "keyword", "graph_local", "graph_global"],
    )

    assert modes
    assert all(m in ["vector", "keyword", "graph_local", "graph_global"] for m in modes)
