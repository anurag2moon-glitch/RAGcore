"""ConceptMapper implementations: map a raw query to relevant concept ids.

EmbeddingConceptMapper is the default and needs no LLM call at all: it
embeds the query and cosine-ranks against every concept's own stored
embedding (Concept.vector, persisted by ragcore.concepts.pipeline.build_concepts
and ragcore.concepts.resolver.EmbeddingConceptResolver). This is not
optional scaffolding kept only for ClaudeConceptMapper to sit on top of — a
prompt cannot list every concept label once a library has thousands of
them, so an embedding shortlist is mandatory regardless of whether an LLM
refinement step is used at all. ClaudeConceptMapper wraps an inner mapper,
takes its shortlist (labels only, never descriptions or vectors — the
explicit cost constraint), and makes one messages.parse call asking which
are actually relevant: a precision filter on the shortlist, never a
replacement for it.

Wiring ConceptRetriever with the default EmbeddingConceptMapper (its own
default) adds zero extra LLM calls to Corpus.ask(); passing a
ClaudeConceptMapper is the explicit opt-in for sharper mapping at the cost
of one call per query.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Sequence

from pydantic import BaseModel

from ragcore.concepts.base import ConceptMapper
from ragcore.concepts.types import Concept
from ragcore.errors import GenerationError
from ragcore.graph.resolver import _cosine

if TYPE_CHECKING:
    from ragcore.embedding.base import Embedder

try:
    import anthropic
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    anthropic = None
else:
    _IMPORT_ERROR = None


class EmbeddingConceptMapper:
    """Default ConceptMapper: no LLM call. See module docstring."""

    def __init__(self, embedder: Embedder, *, shortlist_size: int = 60) -> None:
        self._embedder = embedder
        self._shortlist_size = shortlist_size

    def map(self, query: str, concepts: Sequence[Concept]) -> list[str]:
        candidates = [c for c in concepts if c.vector is not None]
        if not candidates:
            return []
        query_vec = self._embedder.embed_query(query)
        scored = [(c.concept_id, _cosine(query_vec, c.vector)) for c in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [concept_id for concept_id, _score in scored[: self._shortlist_size]]


SYSTEM_PROMPT = (
    "You are given a search query and a numbered list of candidate concept "
    "labels from a book's concept graph. Return the bracketed number(s) of "
    "every candidate that is actually relevant to answering the query "
    "(e.g. 1, 3 — not the text \"[1]\", just the number). Only include a "
    "candidate if it is genuinely relevant, not just superficially similar "
    "wording — it is fine to return none or all of them."
)


class _MappedConcepts(BaseModel):
    relevant_indices: list[int]


class ClaudeConceptMapper:
    """Optional refinement on top of an inner ConceptMapper's shortlist
    (an EmbeddingConceptMapper by default). See module docstring."""

    def __init__(
        self,
        inner: ConceptMapper,
        *,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 1024,
        max_retries: int = 5,
    ) -> None:
        if client is None and anthropic is None:
            raise GenerationError(
                "Claude concept mapping requires the 'generation' extra: "
                "install ragcore[generation]"
            ) from _IMPORT_ERROR

        self.model = model
        self._inner = inner
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def map(self, query: str, concepts: Sequence[Concept]) -> list[str]:
        shortlisted_ids = self._inner.map(query, concepts)
        if not shortlisted_ids:
            return []

        by_id = {c.concept_id: c for c in concepts}
        shortlisted = [by_id[cid] for cid in shortlisted_ids if cid in by_id]
        index_to_concept_id = {i: c.concept_id for i, c in enumerate(shortlisted, start=1)}
        labels_text = "\n".join(f"[{i}] {c.label}" for i, c in enumerate(shortlisted, start=1))
        user_content = f"Query: {query}\n\nCandidate concepts:\n{labels_text}"

        parsed = self._call_with_retry(user_content)
        return [
            index_to_concept_id[i] for i in parsed.relevant_indices if i in index_to_concept_id
        ]

    def _call_with_retry(self, user_content: str) -> _MappedConcepts:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                    output_format=_MappedConcepts,
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
                    raise GenerationError(f"{self.model} concept mapping request rejected: {e}") from e

            if attempt < self._max_retries - 1:
                time.sleep(min(2**attempt, 30))

        raise GenerationError(
            f"{self.model} concept mapping failed after {self._max_retries} attempts: {last_err}"
        ) from last_err
