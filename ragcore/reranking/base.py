"""Reranker protocol: scores (query, document) pairs directly, rather than
relying on embedding cosine similarity or MMR alone — a cross-encoder-style
signal. See ragcore.retrieval.cross_encoder.CrossEncoderReranker for the
Retriever decorator that consumes this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class RerankHit:
    index: int
    """Index into the `documents` list this Reranker.rerank() call was given."""

    relevance_score: float


@runtime_checkable
class Reranker(Protocol):
    model: str

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> list[RerankHit]:
        """Scores every document against `query`, returning hits sorted by
        `relevance_score` descending. `top_n`, when given, is a hint the
        backend may use to only return the top N — a caller that wants
        every document scored (e.g. to decide what to keep itself) should
        leave it as None."""
        ...
