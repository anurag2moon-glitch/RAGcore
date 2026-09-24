"""Reranker implementation using Voyage AI's rerank endpoint
(client.rerank()) — the same `voyageai` package already required for
embeddings (the `voyage` extra), so this is the default cross-encoder
backend with zero new dependency, rather than a local
sentence-transformers/torch model.

Retry convention deliberately mirrors ragcore.embedding.voyage.VoyageContextEmbedder,
not the Claude components' finer-grained retryable/non-retryable split:
Voyage-backed components in this codebase retry broadly and let max_retries
exhaustion be the failure signal, rather than distinguishing voyageai's own
exception hierarchy call by call.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from ragcore.errors import GenerationError
from ragcore.reranking.base import RerankHit

try:
    import voyageai
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    voyageai = None
else:
    _IMPORT_ERROR = None


class VoyageReranker:
    """Reranker implementation using Voyage AI's rerank-2.5 by default."""

    def __init__(
        self,
        *,
        model: str = "rerank-2.5",
        api_key: str | None = None,
        client: Any | None = None,
        max_retries: int = 5,
    ) -> None:
        if client is None and voyageai is None:
            raise GenerationError(
                "Voyage reranking requires the 'voyage' extra: install ragcore[voyage]"
            ) from _IMPORT_ERROR

        self.model = model
        self._max_retries = max_retries
        self._client = client or (
            voyageai.Client(api_key=api_key) if api_key else voyageai.Client()
        )

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> list[RerankHit]:
        if not documents:
            return []

        response = self._call_with_retry(query, list(documents), top_n)
        return [
            RerankHit(index=r.index, relevance_score=r.relevance_score) for r in response.results
        ]

    def _call_with_retry(self, query: str, documents: list[str], top_n: int | None) -> Any:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return self._client.rerank(query, documents, model=self.model, top_k=top_n)
            except Exception as e:  # noqa: BLE001 - retried, then wrapped below
                last_err = e
                if attempt == self._max_retries - 1:
                    break
                time.sleep(min(2**attempt, 30))

        raise GenerationError(
            f"{self.model} reranking failed after {self._max_retries} attempts: {last_err}"
        ) from last_err
