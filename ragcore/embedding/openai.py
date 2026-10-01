"""Embedder backed by OpenAI's embeddings endpoint (text-embedding-3-*
and compatible models) — the second Embedder implementation, proving the
protocol is genuinely swappable rather than swappable in theory only (see
docs/adr/0003-second-embedding-provider.md for why OpenAI was picked over
a local sentence-transformers/BGE model as the second provider).

Two real, load-bearing differences from VoyageContextEmbedder, both
stated here rather than papered over:

1. **No contextualization.** OpenAI's embeddings endpoint has no concept
   of a "group" of inputs embedded jointly — every input string is
   embedded independently of every other. `embed_groups()` still honors
   the Embedder protocol's shape (one vector list per input group, same
   order), but internally flattens every group's texts into one stream of
   independent embedding requests and reassembles the grouping after —
   exactly the "a backend without contextualization support just embeds
   each chunk in its group independently" case the protocol's own
   docstring (ragcore/embedding/base.py) names. A consequence: unlike
   Voyage, nothing here requires a whole group to fit in one request —
   batching only has to respect the flat per-request input/token limits.
2. **No input_type asymmetry.** Voyage (and Cohere) distinguish a
   "document" embedding from a "query" embedding for the same text, which
   measurably improves retrieval for models trained that way.
   text-embedding-3-* was not trained with that asymmetry, so
   `input_type` is accepted (to satisfy the Embedder protocol) but has no
   effect here — silently accepting and ignoring it, rather than raising,
   since a caller swapping Voyage for OpenAI shouldn't have to delete the
   `input_type=` argument at every call site for this to work.

Batching, retry, and rate-limit-pacing structure deliberately mirrors
VoyageContextEmbedder's, so the two feel like genuine interchangeable
peers under the same protocol rather than two unrelated one-off
implementations.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Sequence

from ragcore.chunking.tokenizer import count_tokens
from ragcore.embedding.cost import UsageTracker
from ragcore.errors import EmbeddingError
from ragcore.types import Vector

try:
    import openai
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    openai = None
else:
    _IMPORT_ERROR = None

# Best-effort list-price snapshot (USD per 1M input tokens), not a live
# lookup — see ragcore/embedding/cost.py's module docstring. A model not
# listed here gets no default price (UsageTracker.estimated_cost_usd stays
# None) rather than a guessed number.
_KNOWN_PRICES_PER_MILLION_TOKENS_USD = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


class OpenAIEmbedder:
    """Embedder implementation for OpenAI's text-embedding-3-small/large
    (or any future model with the same flat-input request shape)."""

    MAX_INPUTS_PER_REQUEST = 2_048

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        api_key: str | None = None,
        client: Any | None = None,
        max_retries: int = 5,
        max_tokens_per_request: int = 250_000,
        min_request_interval: float = 0.0,
        price_per_million_tokens_usd: float | None = None,
    ) -> None:
        if client is None and openai is None:
            raise EmbeddingError(
                "OpenAI embedding requires the 'openai' extra: install ragcore[openai]"
            ) from _IMPORT_ERROR

        self.name = model
        self.dim = dim
        self._max_retries = max_retries
        self._max_tokens_per_request = max_tokens_per_request
        self._min_request_interval = min_request_interval
        self._last_request_at: float | None = None
        self._client = client or (
            openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
        )
        self.usage = UsageTracker(
            price_per_million_tokens_usd=(
                price_per_million_tokens_usd
                if price_per_million_tokens_usd is not None
                else _KNOWN_PRICES_PER_MILLION_TOKENS_USD.get(model)
            )
        )

    # -- Embedder protocol -------------------------------------------------

    def embed_groups(
        self,
        groups: Sequence[Sequence[str]],
        *,
        input_type: Literal["document", "query", None] = "document",
    ) -> list[list[Vector]]:
        flat: list[str] = []
        owner: list[int] = []  # flat[i] belongs to groups[owner[i]]
        for group_index, group in enumerate(groups):
            for text in group:
                flat.append(text)
                owner.append(group_index)

        if not flat:
            return [[] for _ in groups]

        vectors_flat: list[Vector | None] = [None] * len(flat)
        for indices, batch in self._batch(flat):
            batch_tokens = sum(count_tokens(t) for t in batch)
            try:
                response = self._call_with_retry(batch)
            except EmbeddingError:
                self.usage.record_failure()
                raise
            self.usage.record_success(batch_tokens)
            for local_idx, vector in enumerate(self._extract(response)):
                vectors_flat[indices[local_idx]] = vector

        out: list[list[Vector]] = [[] for _ in groups]
        for flat_idx, group_index in enumerate(owner):
            out[group_index].append(vectors_flat[flat_idx])  # type: ignore[arg-type]
        return out

    def embed_query(self, text: str) -> Vector:
        return self.embed_groups([[text]], input_type="query")[0][0]

    # -- batching ------------------------------------------------------------

    def _batch(self, texts: list[str]) -> list[tuple[list[int], list[str]]]:
        batches: list[tuple[list[int], list[str]]] = []
        idx_buf: list[int] = []
        text_buf: list[str] = []
        token_count = 0

        for i, text in enumerate(texts):
            t = count_tokens(text)
            if t > self._max_tokens_per_request:
                raise EmbeddingError(
                    f"input {i} has {t} tokens, exceeding max_tokens_per_request "
                    f"({self._max_tokens_per_request}) for {self.name}; raise "
                    f"max_tokens_per_request or shrink the chunk producing it"
                )
            would_overflow_inputs = len(text_buf) >= self.MAX_INPUTS_PER_REQUEST
            would_overflow_tokens = token_count + t > self._max_tokens_per_request
            if text_buf and (would_overflow_inputs or would_overflow_tokens):
                batches.append((idx_buf, text_buf))
                idx_buf, text_buf, token_count = [], [], 0
            idx_buf.append(i)
            text_buf.append(text)
            token_count += t

        if text_buf:
            batches.append((idx_buf, text_buf))
        return batches

    # -- request/response ------------------------------------------------

    def _wait_for_rate_limit(self) -> None:
        if self._min_request_interval <= 0 or self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self._min_request_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _call_with_retry(self, batch: list[str]) -> Any:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            self._wait_for_rate_limit()
            self._last_request_at = time.monotonic()
            try:
                return self._client.embeddings.create(
                    input=batch, model=self.name, dimensions=self.dim
                )
            except Exception as e:  # noqa: BLE001 - retried, then wrapped below
                last_err = e
                if attempt == self._max_retries - 1:
                    break
                time.sleep(min(2**attempt, 30))
        raise EmbeddingError(
            f"{self.name} embedding failed after {self._max_retries} attempts: {last_err}"
        ) from last_err

    @staticmethod
    def _extract(response: Any) -> list[Vector]:
        return [item.embedding for item in response.data]
