"""HierarchySummarizer implementation backed by Claude, using structured
outputs (Pydantic `output_format`, via `client.messages.parse()`) — the same
call shape as every other Claude-backed component in this codebase.

Unlike entity/concept extraction, no citation remapping is needed here: a
node's `child_ids`/`source_chunk_ids` are already known and passed in by the
caller (ragcore.hierarchy.pipeline.build_hierarchy), so the model is never
asked to cite which numbered part it drew from — only to synthesize a
summary from the numbered parts it's shown. `label` is likewise never
asked of the model: it's real structural metadata (chapter_title, section
name, source title) already available from ChunkMeta/SourceInfo, and a real
label beats a model's guess.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Sequence

from pydantic import BaseModel

from ragcore.errors import GenerationError
from ragcore.hierarchy.types import HierarchyLevel, HierarchyNode

try:
    import anthropic
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    anthropic = None
else:
    _IMPORT_ERROR = None

PROMPT_VERSION = "hierarchy-v1"

SYSTEM_PROMPT = (
    "You write a concise summary of one level of a book's structure — a "
    "section, a chapter, or the whole book — from its constituent parts, "
    "each shown numbered in brackets, like [1]. At the section level, the "
    "parts are that section's own paragraphs; at the chapter level, "
    "they're already-written summaries of that chapter's sections (or, if "
    "the chapter has no distinct sections, its own paragraphs directly); "
    "at the book level, they're already-written chapter summaries.\n\n"
    "Write a summary (a few sentences, proportional to how much material "
    "it covers) capturing what this level as a whole discusses. Ground it "
    "in the given parts — do not invent content, and do not simply "
    "concatenate the parts verbatim."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _HierarchySummaryResult(BaseModel):
    summary: str


class ClaudeHierarchySummarizer:
    """HierarchySummarizer implementation using Claude (claude-opus-5 by
    default)."""

    def __init__(
        self,
        *,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 4096,
        max_retries: int = 5,
    ) -> None:
        if client is None and anthropic is None:
            raise GenerationError(
                "Claude hierarchy summarization requires the 'generation' extra: "
                "install ragcore[generation]"
            ) from _IMPORT_ERROR

        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def summarize(
        self,
        *,
        node_id: str,
        level: HierarchyLevel,
        source_id: str,
        label: str,
        child_texts: Sequence[str],
        child_ids: Sequence[str],
        source_chunk_ids: Sequence[str],
    ) -> HierarchyNode:
        formatted = "\n\n".join(f"[{i}]\n{t}" for i, t in enumerate(child_texts, start=1))
        user_content = f"Level: {level}\nLabel: {label}\n\nParts:\n{formatted}"

        parsed = self._call_with_retry(user_content)

        return HierarchyNode(
            node_id=node_id,
            level=level,
            source_id=source_id,
            label=label,
            summary=parsed.summary,
            child_ids=tuple(child_ids),
            source_chunk_ids=tuple(source_chunk_ids),
            model=self.model,
            prompt_version=PROMPT_VERSION,
            created_at=_now(),
        )

    def _call_with_retry(self, user_content: str) -> _HierarchySummaryResult:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                    output_format=_HierarchySummaryResult,
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
                    raise GenerationError(f"{self.model} hierarchy summarization request rejected: {e}") from e

            if attempt < self._max_retries - 1:
                time.sleep(min(2**attempt, 30))

        raise GenerationError(
            f"{self.model} hierarchy summarization failed after {self._max_retries} attempts: {last_err}"
        ) from last_err
