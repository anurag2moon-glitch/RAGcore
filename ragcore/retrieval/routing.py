"""RoutedRetriever — a query router that picks vector, keyword,
graph-local, graph-global, or a combination, per query
(docs/ARCHITECTURE.md Phase 3 gap list item 14). The default stack
(ragcore/retrieval/stack.py) is static once built: every query runs
through the exact same always-on chain of layers. This module adds an
*alternative* composition point — a dict of named, independently-built
Retrievers, with the choice of which one(s) to actually call made fresh
per query by a QueryRouter — rather than changing `default_stack()`
itself, since "always run everything" and "pick what this query needs"
are two different, both legitimate strategies a caller chooses between,
not one replacing the other.

When a router picks more than one mode, their result sets are fused via
`ragcore.retrieval.fusion.rrf_merge` — the same reasoning
ragcore.retrieval.query_expansion.QueryExpansionRetriever already states
for fusing query variants applies here for fusing modes: a keyword-mode
result's BM25-derived score isn't on the same scale as a vector-mode
result's cosine score, so rank (not raw score) is what's comparable
across modes.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel

from ragcore.errors import GenerationError
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
class QueryRouter(Protocol):
    def route(self, query: str, available_modes: Sequence[str]) -> list[str]:
        """Which of `available_modes` to retrieve with for this query — one
        name, or several for a combination. A name not in `available_modes`
        is ignored by the caller (RoutedRetriever), so a router is free to
        propose one it isn't fully sure is offered; an empty or
        all-unrecognized result falls back to the caller's own default."""
        ...


_MODE_DESCRIPTIONS = {
    "vector": "broad semantic/conceptual similarity search — the default "
    "choice for a question about ideas, themes, or meaning rather than "
    "exact wording",
    "keyword": "exact phrase, proper noun, or specific term lookup — best "
    "when the query names something specific that should match "
    "verbatim-or-close source text",
    "graph_local": "a question about a specific named entity's direct "
    "relationships, attributes, or immediate context (who/what is "
    "connected to X)",
    "graph_global": "a broad, thematic, or corpus-wide question better "
    "answered by cross-cutting community/theme summaries than by any "
    "single passage",
    "drift": "a question that needs both: broad thematic/community-level "
    "grounding AND specific entity-level detail together (see "
    "ragcore.retrieval.drift.DriftRetriever) — prefer this single mode "
    "over picking graph_local and graph_global separately when a query "
    "clearly needs both at once",
}


_ROUTING_INTRO = (
    "You route a search query to one or more retrieval modes for a "
    "book corpus. Pick every mode from the list below that would "
    "genuinely help answer this specific query well — one mode for a "
    "query that clearly fits one category, several for a query that "
    "needs more than one angle. Only pick from modes that are listed "
    "as available; never invent a mode name.\n\nAvailable modes:"
)


def _build_system_prompt(available_modes: Sequence[str]) -> str:
    lines = [_ROUTING_INTRO]
    for mode in available_modes:
        description = _MODE_DESCRIPTIONS.get(mode, "a retrieval mode specific to this corpus")
        lines.append(f"- {mode}: {description}")
    return "\n".join(lines)


class _RoutingResult(BaseModel):
    modes: list[str]
    reasoning: str


class ClaudeQueryRouter:
    """QueryRouter implementation using Claude (claude-opus-5 by
    default)."""

    def __init__(
        self,
        *,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 512,
        max_retries: int = 5,
    ) -> None:
        if client is None and anthropic is None:
            raise GenerationError(
                "Claude query routing requires the 'generation' extra: "
                "install ragcore[generation]"
            ) from _IMPORT_ERROR

        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def route(self, query: str, available_modes: Sequence[str]) -> list[str]:
        if not available_modes:
            return []
        if len(available_modes) == 1:
            return list(available_modes)  # nothing to decide, no call needed

        system_prompt = _build_system_prompt(available_modes)
        parsed = self._call_with_retry(query, system_prompt)
        return [m for m in parsed.modes if m in available_modes]

    def _call_with_retry(self, query: str, system_prompt: str) -> _RoutingResult:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": f"Query: {query}"}],
                    output_format=_RoutingResult,
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
                    raise GenerationError(f"{self.model} query routing request rejected: {e}") from e

            if attempt < self._max_retries - 1:
                time.sleep(min(2**attempt, 30))

        raise GenerationError(
            f"{self.model} query routing failed after {self._max_retries} attempts: {last_err}"
        ) from last_err


class RoutedRetriever:
    """Picks one or more named Retrievers per query via `router.route()`,
    fusing their results via RRF when more than one is chosen. See module
    docstring for the full rationale.

    `default_modes` (default: every mode) is used when the router returns
    nothing recognized — not an error case, just a signal to fall back to
    a safe default rather than retrieving nothing at all.
    """

    def __init__(
        self,
        retrievers: dict[str, Retriever],
        router: QueryRouter,
        *,
        rrf_k: int = 60,
        default_modes: Sequence[str] | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("RoutedRetriever requires at least one named retriever")
        self._retrievers = retrievers
        self._router = router
        self._rrf_k = rrf_k
        self._default_modes = list(default_modes) if default_modes is not None else list(retrievers)
        self.last_route: tuple[str, ...] = ()

    def retrieve(
        self, query: str, *, k: int = 10, filters: Filter | None = None
    ) -> list[Retrieved]:
        if k <= 0:
            self.last_route = ()
            return []

        modes = self._router.route(query, list(self._retrievers))
        modes = [m for m in modes if m in self._retrievers] or list(self._default_modes)
        self.last_route = tuple(modes)

        labeled = [(mode, self._retrievers[mode].retrieve(query, k=k, filters=filters)) for mode in modes]
        return rrf_merge(labeled, k=k, rrf_k=self._rrf_k, extra_key="routed_modes")
