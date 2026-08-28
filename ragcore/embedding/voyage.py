"""Embedder backed by Voyage AI's contextualized chunk embeddings
(`voyage-context-4`).

Groups are sent pre-chunked (`inputs: List[List[str]]`) rather than relying
on Voyage's auto-chunking — see the design notes in the project plan: an
auto-chunked response returns chunk text but no character offsets into the
source, which would make it impossible to recover page/section provenance.
Sending pre-chunked groups keeps contextualization (each group's chunks are
embedded with awareness of each other) while keeping offsets intact.

Batches requests to stay under the API's per-request limits: 1,000 inputs
(ChunkGroups), 16,000 chunks total, and `max_tokens_per_request` tokens
total (default 100,000 — a conservative default well under Voyage's own
120K-token ceiling, and small enough to also respect lower account-tier
rate limits like the unfunded-account default of 10K tokens/minute, which
a token-blind batcher can blow through in a single request on any book of
meaningful length). A single ChunkGroup larger than either the chunk or
token limit can't be split without breaking its contextualization window,
so that raises EmbeddingError instead of silently truncating — the fix is
a smaller `max_group_tokens` on the Chunker, not truncation here.

`min_request_interval` optionally paces requests (sleeps between them) to
stay under a request-per-minute cap — irrelevant on a funded account, but
without it a free/unfunded Voyage account (3 RPM) fails outright even with
correctly token-sized batches.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Sequence

from ragcore.chunking.tokenizer import count_tokens
from ragcore.errors import EmbeddingError
from ragcore.types import Vector

try:
    import voyageai
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    voyageai = None
else:
    _IMPORT_ERROR = None

OutputDtype = Literal["float", "int8", "uint8", "binary", "ubinary"]


class VoyageContextEmbedder:
    """Embedder implementation for voyage-context-4 (or any future
    voyage-context-* model with the same request shape)."""

    MAX_INPUTS_PER_REQUEST = 1_000
    MAX_CHUNKS_PER_REQUEST = 16_000

    def __init__(
        self,
        *,
        model: str = "voyage-context-4",
        output_dimension: int = 1024,
        output_dtype: OutputDtype = "float",
        api_key: str | None = None,
        client: Any | None = None,
        max_retries: int = 5,
        max_tokens_per_request: int = 100_000,
        min_request_interval: float = 0.0,
    ) -> None:
        if client is None and voyageai is None:
            raise EmbeddingError(
                "voyage-context-4 embedding requires the 'voyage' extra: "
                "install ragcore[voyage]"
            ) from _IMPORT_ERROR

        self.name = model
        self.dim = output_dimension
        self._output_dtype = output_dtype
        self._max_retries = max_retries
        self._max_tokens_per_request = max_tokens_per_request
        self._min_request_interval = min_request_interval
        self._last_request_at: float | None = None
        self._client = client or (
            voyageai.Client(api_key=api_key) if api_key else voyageai.Client()
        )

    # -- Embedder protocol -------------------------------------------------

    def embed_groups(
        self,
        groups: Sequence[Sequence[str]],
        *,
        input_type: Literal["document", "query", None] = "document",
    ) -> list[list[Vector]]:
        results: list[list[Vector] | None] = [None] * len(groups)
        for indices, batch in self._batch(groups):
            response = self._call_with_retry(batch, input_type)
            for local_idx, group_embeddings in enumerate(self._extract(response)):
                results[indices[local_idx]] = group_embeddings
        # every index was assigned exactly once by _batch's partition of range(len(groups))
        return results  # type: ignore[return-value]

    def embed_query(self, text: str) -> Vector:
        return self.embed_groups([[text]], input_type="query")[0][0]

    # -- batching ------------------------------------------------------------

    def _batch(
        self, groups: Sequence[Sequence[str]]
    ) -> list[tuple[list[int], list[Sequence[str]]]]:
        batches: list[tuple[list[int], list[Sequence[str]]]] = []
        idx_buf: list[int] = []
        group_buf: list[Sequence[str]] = []
        chunk_count = 0
        token_count = 0

        for i, group in enumerate(groups):
            n = len(group)
            if n > self.MAX_CHUNKS_PER_REQUEST:
                raise EmbeddingError(
                    f"group {i} has {n} chunks, exceeding the "
                    f"{self.MAX_CHUNKS_PER_REQUEST}-chunk-per-request limit for "
                    f"{self.name}; use a Chunker with a smaller max_group_tokens"
                )
            group_tokens = sum(count_tokens(t) for t in group)
            if group_tokens > self._max_tokens_per_request:
                raise EmbeddingError(
                    f"group {i} has {group_tokens} tokens, exceeding "
                    f"max_tokens_per_request ({self._max_tokens_per_request}) for "
                    f"{self.name}; use a Chunker with a smaller max_group_tokens or "
                    f"raise max_tokens_per_request"
                )
            would_overflow_inputs = len(group_buf) >= self.MAX_INPUTS_PER_REQUEST
            would_overflow_chunks = chunk_count + n > self.MAX_CHUNKS_PER_REQUEST
            would_overflow_tokens = token_count + group_tokens > self._max_tokens_per_request
            if group_buf and (would_overflow_inputs or would_overflow_chunks or would_overflow_tokens):
                batches.append((idx_buf, group_buf))
                idx_buf, group_buf, chunk_count, token_count = [], [], 0, 0
            idx_buf.append(i)
            group_buf.append(group)
            chunk_count += n
            token_count += group_tokens

        if group_buf:
            batches.append((idx_buf, group_buf))
        return batches

    # -- request/response ------------------------------------------------

    def _wait_for_rate_limit(self) -> None:
        if self._min_request_interval <= 0 or self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self._min_request_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _call_with_retry(
        self, batch: list[Sequence[str]], input_type: Literal["document", "query", None]
    ) -> Any:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            self._wait_for_rate_limit()
            self._last_request_at = time.monotonic()
            try:
                return self._client.contextualized_embed(
                    inputs=batch,
                    model=self.name,
                    input_type=input_type,
                    output_dimension=self.dim,
                    output_dtype=self._output_dtype,
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
    def _extract(response: Any) -> list[list[Vector]]:
        """Pull per-group embedding lists out of the SDK response.

        Tries the SDK's object attributes first (`.results[i].embeddings`);
        falls back to the raw JSON response shape
        (`{"data": [{"data": [{"embedding": [...]}]}]}`) so this keeps
        working across voyageai SDK versions that wrap the response
        differently.
        """
        results = getattr(response, "results", None)
        if results is not None:
            return [list(r.embeddings) for r in results]

        data = response["data"] if isinstance(response, dict) else response.data
        out: list[list[Vector]] = []
        for group in data:
            items = group["data"] if isinstance(group, dict) else group.data
            out.append(
                [item["embedding"] if isinstance(item, dict) else item.embedding for item in items]
            )
        return out
