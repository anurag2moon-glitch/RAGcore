"""QueryExpansionRetriever — a standalone query-understanding layer:
multi-query expansion, HyDE, and query decomposition, as a retrieval-layer
primitive independent of ragcore.conversation's history-aware rewriting
(docs/ARCHITECTURE.md Phase 3 gap list item 13).

ragcore.conversation.rewriting.HistoryAwareQueryRewriter solves a
different problem — turning a conversational turn with implicit
references ("what about that") into one self-contained query, using the
transcript as its only input. This module solves query understanding for
a query that's *already* self-contained but could still retrieve better
with more than one search: a HyDE hypothetical passage embeds closer to
an actual answer-containing chunk than the question itself does;
alternative phrasings cover source material using different terminology
than the user's; and a compound question ("how does X relate to Y, and
what did the author later say about Z?") retrieves better as several
targeted searches than one blended one. Both layers compose freely —
HistoryAwareQueryRewriter's output is exactly the kind of self-contained
query this layer then expands.

One Claude call produces all three expansion types together (not three
separate calls) — cost-conscious by construction, matching every other
LLM-backed layer in this codebase (ragcore.concepts.mapper's
shortlist-then-refine, HistoryAwareQueryRewriter's free first-turn case).

QueryExpansionRetriever fuses the original query's results with every
expansion variant's results via reciprocal rank fusion (RRF) — the same
technique and the same reason ragcore.retrieval.hybrid.HybridRetriever
fuses its two channels by RRF rather than raw score: a HyDE passage
embeds in a different region of vector space than a short question does,
so comparing their raw scores directly isn't meaningful, but comparing
*ranks* within each variant's own result list is. RRF decides the
candidate set and final order; the `Retrieved.score` on each result stays
its own best real score from whichever variant found it (never a raw RRF
score), preserving the same score-scale invariant every other layer in
this stack depends on.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from ragcore.errors import GenerationError
from ragcore.model_routing import DEFAULT_ROUTING
from ragcore.retrieval.base import Retriever
from ragcore.retrieval.fusion import rrf_merge
from ragcore.types import Filter, Retrieved

try:
    import anthropic
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    anthropic = None
else:
    _IMPORT_ERROR = None


@runtime_checkable
class QueryTransform(Protocol):
    def expand(self, query: str) -> list[str]:
        """Zero or more additional query strings to retrieve alongside the
        original (the original is always retrieved too — see
        QueryExpansionRetriever.retrieve — so an empty list is a valid,
        inexpensive "nothing to add" response, not an error)."""
        ...


SYSTEM_PROMPT = (
    "You help a retrieval system search a book corpus more effectively for "
    "a given query. Produce up to three kinds of additional search input:\n\n"
    "1. hyde_passage: a short (2-4 sentence) HYPOTHETICAL passage that "
    "would directly answer the query if it appeared verbatim in a source "
    "book — written as if it WERE real source text (not as an answer "
    "addressed to the user, never \"The answer is...\" or similar framing). "
    "This is used only to compute an embedding for similarity search, never "
    "shown to anyone — write naturally, the way a book would actually say "
    "it. Always provide this.\n\n"
    "2. alternative_phrasings: 0 to 3 alternative ways to phrase the same "
    "underlying information need, each covering a different angle or "
    "terminology a source might use instead of the query's own wording. "
    "Leave empty if the query's own phrasing is already about as good as "
    "any alternative would be.\n\n"
    "3. sub_questions: if and only if the query bundles multiple distinct "
    "questions or requests, break it into 2-4 standalone sub-questions, "
    "each answerable independently. Leave empty if the query is already "
    "one focused question."
)


class _ExpansionResult(BaseModel):
    hyde_passage: str
    alternative_phrasings: list[str] = []
    sub_questions: list[str] = []


class ClaudeQueryExpander:
    """QueryTransform implementation using Claude (cheap-tier by default —
    see ragcore.model_routing; expansion, not final-answer synthesis), one
    call producing HyDE + alternative phrasings + sub-questions together."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_ROUTING.cheap,
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 1024,
        max_retries: int = 5,
        max_variants: int = 5,
    ) -> None:
        if client is None and anthropic is None:
            raise GenerationError(
                "Claude query expansion requires the 'generation' extra: "
                "install ragcore[generation]"
            ) from _IMPORT_ERROR

        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._max_variants = max_variants
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def expand(self, query: str) -> list[str]:
        parsed = self._call_with_retry(query)
        variants = [parsed.hyde_passage, *parsed.alternative_phrasings, *parsed.sub_questions]
        # dedupe while preserving order (a paraphrase identical to the
        # original query, or two near-duplicate sub-questions, cost an
        # extra retrieval call for no benefit)
        seen: set[str] = set()
        out: list[str] = []
        for v in variants:
            v = v.strip()
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out[: self._max_variants]

    def _call_with_retry(self, query: str) -> _ExpansionResult:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": f"Query: {query}"}],
                    output_format=_ExpansionResult,
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
                    raise GenerationError(f"{self.model} query expansion request rejected: {e}") from e

            if attempt < self._max_retries - 1:
                time.sleep(min(2**attempt, 30))

        raise GenerationError(
            f"{self.model} query expansion failed after {self._max_retries} attempts: {last_err}"
        ) from last_err


class QueryExpansionRetriever:
    """Decorates a Retriever: retrieves the original query plus every
    `transform.expand(query)` variant, fuses the result sets by RRF. See
    module docstring for the full rationale."""

    def __init__(self, inner: Retriever, transform: QueryTransform, *, rrf_k: int = 60) -> None:
        self._inner = inner
        self._transform = transform
        self._rrf_k = rrf_k

    def retrieve(
        self, query: str, *, k: int = 10, filters: Filter | None = None
    ) -> list[Retrieved]:
        if k <= 0:
            return []

        variants = [query, *self._transform.expand(query)]
        labeled = [(v, self._inner.retrieve(v, k=k, filters=filters)) for v in variants]
        return rrf_merge(labeled, k=k, rrf_k=self._rrf_k, extra_key="query_expansion_variants")
