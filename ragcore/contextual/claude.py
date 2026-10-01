"""ContextWriter implementation backed by Claude, using structured outputs
(Pydantic `output_format`, via `client.messages.parse()`) — the same call
shape as ragcore.graph.claude_extractor.ClaudeEntityExtractor.

Called once per ChunkGroup (chapter) by
ragcore.contextual.pipeline.build_contextual_embeddings, reusing the
numbered-citation scheme from ragcore.graph.claude_extractor._format_chunks
so the model always answers with a bare number, never the real chunk_id.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Sequence

from pydantic import BaseModel

from ragcore.errors import GenerationError
from ragcore.graph.claude_extractor import _format_chunks
from ragcore.model_routing import DEFAULT_ROUTING

if TYPE_CHECKING:
    from ragcore.types import Chunk

try:
    import anthropic
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    anthropic = None
else:
    _IMPORT_ERROR = None

SYSTEM_PROMPT = (
    "You write short context blurbs for chunks of book text, to be "
    "prepended before the chunk is embedded — so a retrieval system can "
    "tell what a passage is about even in isolation, without seeing the "
    "surrounding text. You are given every chunk of one chapter, in order, "
    "each labeled with a number in brackets, like [1].\n\n"
    "For each chunk, write ONE short blurb (1-2 sentences) noting: which "
    "section or chapter it's from (use the actual heading/title language "
    "visible in the surrounding text if apparent, not a generic label), "
    "and what this specific chunk discusses in relation to what comes "
    "before and after it in the chapter. This is situating metadata, not a "
    "summary of the whole chapter — keep it grounded in this one chunk.\n\n"
    "For each chunk, give the bracketed number it corresponds to (e.g. 1, "
    "not the text \"[1]\", just the number) alongside its blurb. Give "
    "exactly one blurb per chunk shown."
)


class _Blurb(BaseModel):
    chunk_index: int
    blurb: str


class _BlurbResult(BaseModel):
    blurbs: list[_Blurb]


class ClaudeContextWriter:
    """ContextWriter implementation using Claude (cheap-tier by default —
    see ragcore.model_routing; a short blurb, not final-answer synthesis)."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_ROUTING.cheap,
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 4096,
        max_retries: int = 5,
    ) -> None:
        if client is None and anthropic is None:
            raise GenerationError(
                "Claude context-writing requires the 'generation' extra: "
                "install ragcore[generation]"
            ) from _IMPORT_ERROR

        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def write_blurbs(self, chunks: Sequence[Chunk]) -> list[str]:
        if not chunks:
            return []

        formatted, _index_to_chunk_id = _format_chunks(chunks)
        parsed = self._call_with_retry(formatted)

        # A missing index (the model skipped a chunk) degrades to an empty
        # blurb rather than crashing — the chunk still embeds fine on its
        # own text, just without the situating context.
        blurb_by_index = {b.chunk_index: b.blurb for b in parsed.blurbs}
        return [blurb_by_index.get(i, "") for i in range(1, len(chunks) + 1)]

    def _call_with_retry(self, user_content: str) -> _BlurbResult:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                    output_format=_BlurbResult,
                )
                return response.parsed_output
            except anthropic.RateLimitError as e:
                last_err = e
            except anthropic.APIConnectionError as e:
                last_err = e
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    last_err = e
                else:
                    raise GenerationError(f"{self.model} context-writing request rejected: {e}") from e

            if attempt < self._max_retries - 1:
                time.sleep(min(2**attempt, 30))

        raise GenerationError(
            f"{self.model} context-writing failed after {self._max_retries} attempts: {last_err}"
        ) from last_err
