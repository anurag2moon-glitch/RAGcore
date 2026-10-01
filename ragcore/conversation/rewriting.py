"""HistoryAwareQueryRewriter: the query-rewriting step that runs strictly
BEFORE the existing retrieval pipeline, using structured outputs
(`client.messages.parse()`) — the same call shape every other Claude-backed
component in this codebase uses.

Cost-conscious by construction, matching ragcore.concepts.mapper's
shortlist-then-refine discipline: a transcript of length <= 1 (the very
first turn of a session, or a debate's seeded topic turn with no reply yet)
needs no rewriting at all — there is nothing prior to resolve pronouns or
ellipsis against — so that case returns the turn's own text unchanged with
no API call.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from pydantic import BaseModel

from ragcore.conversation.types import Turn
from ragcore.errors import GenerationError
from ragcore.model_routing import DEFAULT_ROUTING

try:
    import anthropic
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    anthropic = None
else:
    _IMPORT_ERROR = None

PROMPT_VERSION = "rewrite-v1"

SYSTEM_PROMPT = (
    "You rewrite the next turn of a conversation or debate transcript into "
    "one self-contained search query, for a retrieval system that has no "
    "memory of the transcript itself.\n\n"
    "You are given the transcript so far, each line labeled with its "
    "speaker, and told who is about to respond next. Write a single query "
    "that speaker's own retrieval system would need, to find the material "
    "it needs to respond well — resolving pronouns, ellipsis, and implicit "
    "references (\"what about that\", \"when I feel guilty about it\") "
    "against the transcript so the query stands completely on its own. Do "
    "not answer the question yourself, and do not invent details the "
    "transcript doesn't support — only make the existing request "
    "self-contained."
)


class _RewrittenQuery(BaseModel):
    query: str


def _format_transcript(transcript: Sequence[Turn]) -> str:
    return "\n".join(f"{t.speaker}: {t.text}" for t in transcript)


class HistoryAwareQueryRewriter:
    """QueryRewriter implementation using Claude (cheap-tier by default —
    see ragcore.model_routing; rewriting, not final-answer synthesis)."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_ROUTING.cheap,
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 512,
        max_retries: int = 5,
    ) -> None:
        if client is None and anthropic is None:
            raise GenerationError(
                "Claude query rewriting requires the 'generation' extra: "
                "install ragcore[generation]"
            ) from _IMPORT_ERROR

        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def rewrite(self, *, transcript: Sequence[Turn], speaking_as: str) -> str:
        if not transcript:
            return ""
        if len(transcript) <= 1:
            return transcript[-1].text

        user_content = (
            f"{speaking_as} is about to respond next.\n\n"
            f"Transcript so far:\n{_format_transcript(transcript)}"
        )
        parsed = self._call_with_retry(user_content)
        return parsed.query

    def _call_with_retry(self, user_content: str) -> _RewrittenQuery:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                    output_format=_RewrittenQuery,
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
                    raise GenerationError(f"{self.model} query rewriting request rejected: {e}") from e

            if attempt < self._max_retries - 1:
                time.sleep(min(2**attempt, 30))

        raise GenerationError(
            f"{self.model} query rewriting failed after {self._max_retries} attempts: {last_err}"
        ) from last_err
