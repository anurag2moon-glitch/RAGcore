"""Generator implementation backed by the Claude API (Messages endpoint).

Builds a grounded prompt from Retrieved chunks — each tagged with its
chapter/section/page so the model can cite structure — instructs the model
to answer only from that context, and returns an Answer. Uses the official
`anthropic` Python SDK (>=1.0); nothing here talks HTTP directly.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Sequence

from ragcore.errors import GenerationError
from ragcore.generation.base import Answer
from ragcore.generation.context_packer import pack_context
from ragcore.prompt_caching import cached_system
from ragcore.types import Retrieved

if TYPE_CHECKING:
    from ragcore.corpus import Corpus

try:
    import anthropic
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    anthropic = None
else:
    _IMPORT_ERROR = None

DEFAULT_SYSTEM_PROMPT = (
    "You are answering questions using only the provided source excerpts from a "
    "book. Each excerpt is numbered and tagged with its chapter/section/page. "
    "Answer using only this context — if the context doesn't contain enough "
    "information to answer, say so explicitly rather than guessing or using "
    "outside knowledge. Where helpful, reference which excerpt number(s) "
    "support each part of your answer."
)


def _format_context(context: Sequence[Retrieved]) -> str:
    parts: list[str] = []
    for i, r in enumerate(context, start=1):
        m = r.chunk.meta
        loc = m.chapter_title or "unknown location"
        if m.section:
            loc += f", {m.section}"
        if m.page_start is not None:
            loc += f", p.{m.page_start}"
        parts.append(f"[{i}] ({loc})\n{r.chunk.text}")
    return "\n\n".join(parts)


class ClaudeGenerator:
    """Generator implementation using Claude (claude-opus-5 by default).

    `context_max_tokens` is optional (default None) and opt-in: see
    ragcore.generation.context_packer. Leaving it unset keeps `generate()`'s
    original behavior exactly — every retrieved chunk goes into the
    prompt and `Answer.citations` unchanged, no token accounting at all.
    Setting it packs `context` through `pack_context()` first (dedupe,
    optional neighbor expansion via `corpus`/`expand_neighbors`, token
    budgeting, document-order presentation) and citations become only
    what actually made the prompt.

    `enable_prompt_caching` (default False, opt-in): see
    ragcore.prompt_caching. The system prompt is identical on every call
    this Generator ever makes, so marking it cacheable lets a corpus that
    calls `ask()`/`generate()` repeatedly skip reprocessing it past the
    first call. Default False keeps `system=` a plain string, the exact
    request shape every existing caller already gets.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        client: Any | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        max_retries: int = 5,
        context_max_tokens: int | None = None,
        context_reserve_tokens: int = 0,
        corpus: Corpus | None = None,
        expand_neighbors: int = 0,
        enable_prompt_caching: bool = False,
    ) -> None:
        if client is None and anthropic is None:
            raise GenerationError(
                "Claude generation requires the 'generation' extra: "
                "install ragcore[generation]"
            ) from _IMPORT_ERROR

        self.model = model
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._context_max_tokens = context_max_tokens
        self._context_reserve_tokens = context_reserve_tokens
        self._corpus = corpus
        self._expand_neighbors = expand_neighbors
        self._enable_prompt_caching = enable_prompt_caching
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def generate(self, query: str, context: Sequence[Retrieved]) -> Answer:
        if self._context_max_tokens is not None:
            packed = pack_context(
                context,
                max_tokens=self._context_max_tokens,
                reserve_tokens=self._context_reserve_tokens,
                corpus=self._corpus,
                expand_neighbors=self._expand_neighbors,
            )
            context = packed.kept

        user_content = f"{_format_context(context)}\n\nQuestion: {query}"
        response = self._call_with_retry(user_content)

        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            raise GenerationError(f"{self.model} refused to answer (category={category!r})")

        text = "".join(block.text for block in response.content if block.type == "text")
        return Answer(text=text, citations=list(context))

    def _call_with_retry(self, user_content: str) -> Any:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return self._client.messages.create(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=(
                        cached_system(self._system_prompt)
                        if self._enable_prompt_caching
                        else self._system_prompt
                    ),
                    messages=[{"role": "user", "content": user_content}],
                )
            except anthropic.RateLimitError as e:
                last_err = e
            except anthropic.APIConnectionError as e:
                last_err = e
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    last_err = e
                else:
                    # non-retryable client error (bad request, auth, ...)
                    raise GenerationError(f"{self.model} generation request rejected: {e}") from e

            if attempt < self._max_retries - 1:
                time.sleep(min(2**attempt, 30))

        raise GenerationError(
            f"{self.model} generation failed after {self._max_retries} attempts: {last_err}"
        ) from last_err
