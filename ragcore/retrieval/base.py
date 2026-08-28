"""Retriever protocol: the composition point for query -> results.

Corpus.retrieve() delegates to a Retriever instance. VectorRetriever is the
only implementation until the graph layer lands; GraphRetriever will then
wrap a Retriever with the same protocol (decorator, not a different
interface), so `Corpus.retriever = GraphRetriever(corpus.retriever, ...)` is
the entire integration step for a consuming project — retrieve() call sites
never change.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragcore.types import Filter, Retrieved


@runtime_checkable
class Retriever(Protocol):
    def retrieve(
        self, query: str, *, k: int = 10, filters: Filter | None = None
    ) -> list[Retrieved]: ...
