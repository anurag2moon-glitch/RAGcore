"""Live smoke test against the real OpenAI embeddings API. Excluded by
default (see pyproject.toml `addopts = "-m 'not live'"`); run explicitly
with `pytest -m live`. Requires OPENAI_API_KEY in the environment.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_embed_groups_returns_real_vectors_and_flattens_correctly() -> None:
    from ragcore.embedding.openai import OpenAIEmbedder

    embedder = OpenAIEmbedder(dim=256)
    groups = [
        ["The mitochondria is the powerhouse of the cell.", "It produces ATP."],
        ["Attention is all you need.", "Transformers replaced recurrence."],
    ]

    vectors = embedder.embed_groups(groups, input_type="document")

    assert len(vectors) == 2
    assert len(vectors[0]) == 2
    assert len(vectors[1]) == 2
    assert all(len(v) == 256 for group in vectors for v in group)

    query_vector = embedder.embed_query("What produces energy in a cell?")
    assert len(query_vector) == 256
