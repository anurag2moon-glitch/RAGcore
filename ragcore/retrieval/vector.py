"""VectorRetriever — the default (and, until the graph layer lands, only)
Retriever implementation. Embeds the query, delegates the similarity +
filter search to the VectorStore.
"""

from __future__ import annotations

from ragcore.embedding.base import Embedder
from ragcore.stores.base import VectorStore
from ragcore.types import Filter, Retrieved


class VectorRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(
        self, query: str, *, k: int = 10, filters: Filter | None = None
    ) -> list[Retrieved]:
        vector = self._embedder.embed_query(query)
        return self._store.query(vector, k, filters)
