"""A deterministic, offline Embedder + Generator pair - no network,
no API key - for the CI integration test (docs/adr/0022) against a
*real* running Compose stack, where no VOYAGE_API_KEY/ANTHROPIC_API_KEY
secret is assumed to exist (this repo's existing CI never runs
`pytest -m live` for the same reason). Activated only via
RAGCORE_USE_FAKE_MODELS=1 (app.settings.Settings.ragcore_use_fake_models) -
never set this in a real deployment; every route would appear to work
but return semantically meaningless answers.

Unlike tests/fakes.py's FakeEmbedder (every text -> the same constant
vector, fine for in-process tests that never check *which* chunk was
retrieved), this uses a real hashing-trick bag-of-words embedding so
two texts sharing vocabulary score closer than two that don't - a CI
integration test asserting "the retrieved chunk is actually the one
about X" needs that; a constant embedding would make retrieval order
arbitrary.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Literal

from ragcore.generation.base import Answer
from ragcore.types import Retrieved, Vector


class HashingEmbedder:
    """Deterministic, dependency-free bag-of-words embedding via the
    hashing trick (each word hashes to one of `dim` buckets,
    L2-normalized) - not semantically meaningful the way a real
    embedding model is, but real cosine similarity between texts that
    share vocabulary, which is all a smoke test needs."""

    dim = 64
    name = "ci-hashing-embedder"

    def _embed_one(self, text: str) -> Vector:
        vec = [0.0] * self.dim
        for word in text.lower().split():
            bucket = int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_query(self, text: str) -> Vector:
        return self._embed_one(text)

    def embed_groups(
        self, groups: Sequence[Sequence[str]], *, input_type: Literal["document", "query"] | None = "document"
    ) -> list[list[Vector]]:
        return [[self._embed_one(text) for text in group] for group in groups]


class EchoCitingGenerator:
    """Never calls out to any LLM - cites every chunk it was given as
    context, verbatim, so /ask's citation field is exercisable end to
    end without a real ANTHROPIC_API_KEY."""

    name = "ci-echo-generator"

    def generate(self, query: str, context: list[Retrieved]) -> Answer:
        return Answer(text=f"[ci-fake answer to {query!r}]", citations=list(context))
