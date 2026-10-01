"""Live smoke test against the real Claude API. Excluded by default (see
pyproject.toml `addopts = "-m 'not live'"`); run explicitly with
`pytest -m live`. Requires ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_claude_query_expander_produces_real_variants() -> None:
    from ragcore.retrieval.query_expansion import ClaudeQueryExpander

    expander = ClaudeQueryExpander()
    variants = expander.expand("What caused the fall of the Roman Empire?")

    assert variants
    assert all(isinstance(v, str) and v.strip() for v in variants)
