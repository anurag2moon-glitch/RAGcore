"""IterativeRetriever — decorates a Retriever with a multi-hop loop: retrieve
once, ask an LLM whether the result is sufficient to answer the original
query, and if not, retrieve again with a refined sub-query, up to a
configurable cap (default 3 hops).

Fits the Retriever protocol's existing single retrieve() signature with no
changes — the loop runs entirely inside one retrieve() call, so
`corpus.retriever = IterativeRetriever(...)` composes exactly like every
other decorator, and Corpus.ask()/Corpus.retrieve() need no changes at all.
Hop provenance is recoverable two ways: from
`retrieved.chunk.meta.extra["hop_index"]`/`["hop_query"]` on every returned
result (stamped the same way Corpus.ingest() already tags chunk metadata via
dataclasses.replace), and, for debugging, from
`IterativeRetriever.last_hops` after a retrieve() call.

Composition point: IterativeRetriever wraps the concept/graph/hybrid stack
(everything that varies its OWN query per call), while cross-encoder
reranking and MMR diversity sit *outside* it, run once over the final
merged multi-hop union — see ragcore/retrieval/cross_encoder.py's module
docstring for why reranking per hop would produce an internally
inconsistent ranking (a cross-encoder score against hop 2's reformulated
sub-query isn't comparable to one against hop 1's original query).
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel

from ragcore.errors import GenerationError
from ragcore.graph.claude_extractor import _format_chunks
from ragcore.model_routing import DEFAULT_ROUTING
from ragcore.retrieval.base import Retriever
from ragcore.types import Filter, Retrieved

try:
    import anthropic
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    anthropic = None
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class Sufficiency:
    sufficient: bool
    next_query: str | None
    reasoning: str


@dataclass(frozen=True)
class HopRecord:
    hop_index: int
    query: str
    retrieved_chunk_ids: tuple[str, ...]
    sufficient: bool
    reasoning: str


@runtime_checkable
class SufficiencyJudge(Protocol):
    model: str

    def assess(
        self, original_query: str, hops_so_far: Sequence[HopRecord], context: Sequence[Retrieved]
    ) -> Sufficiency: ...


SYSTEM_PROMPT = (
    "You judge whether a set of retrieved source excerpts is sufficient to "
    "answer a user's original question. You are given the original "
    "question, a summary of what's been tried so far (previous "
    "hops/sub-queries, if any), and the excerpts retrieved so far, each "
    "numbered in brackets, like [1].\n\n"
    "Decide: is this enough to answer the original question well? If yes, "
    "say so. If not, propose ONE focused follow-up search query that would "
    "help fill the specific gap — a different angle, a missing entity or "
    "detail the current excerpts don't cover, or a more specific "
    "rephrasing. Do not propose a follow-up query that is essentially the "
    "same as the original question or a previous hop's query. Briefly "
    "explain your reasoning either way."
)


class _SufficiencyResult(BaseModel):
    sufficient: bool
    next_query: str | None = None
    reasoning: str


class ClaudeSufficiencyJudge:
    """SufficiencyJudge implementation using Claude (cheap-tier by default
    — see ragcore.model_routing; judging, not final-answer synthesis)."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_ROUTING.cheap,
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 1024,
        max_retries: int = 5,
    ) -> None:
        if client is None and anthropic is None:
            raise GenerationError(
                "Claude sufficiency judging requires the 'generation' extra: "
                "install ragcore[generation]"
            ) from _IMPORT_ERROR

        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def assess(
        self, original_query: str, hops_so_far: Sequence[HopRecord], context: Sequence[Retrieved]
    ) -> Sufficiency:
        formatted, _index_to_chunk_id = _format_chunks([r.chunk for r in context])
        history = (
            "\n".join(f"- hop {h.hop_index}: {h.query!r}" for h in hops_so_far)
            if hops_so_far
            else "(none — this is the first retrieval)"
        )
        user_content = (
            f"Original question: {original_query}\n\n"
            f"Previous hops:\n{history}\n\n"
            f"Retrieved excerpts:\n{formatted}"
        )
        parsed = self._call_with_retry(user_content)
        return Sufficiency(
            sufficient=parsed.sufficient, next_query=parsed.next_query, reasoning=parsed.reasoning
        )

    def _call_with_retry(self, user_content: str) -> _SufficiencyResult:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                    output_format=_SufficiencyResult,
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
                    raise GenerationError(f"{self.model} sufficiency judging request rejected: {e}") from e

            if attempt < self._max_retries - 1:
                time.sleep(min(2**attempt, 30))

        raise GenerationError(
            f"{self.model} sufficiency judging failed after {self._max_retries} attempts: {last_err}"
        ) from last_err


def _stamp_hop(r: Retrieved, hop_index: int, query: str) -> Retrieved:
    extra = {**r.chunk.meta.extra, "hop_index": hop_index, "hop_query": query}
    stamped_chunk = dataclasses.replace(r.chunk, meta=dataclasses.replace(r.chunk.meta, extra=extra))
    return dataclasses.replace(r, chunk=stamped_chunk)


class IterativeRetriever:
    def __init__(
        self, inner: Retriever, judge: SufficiencyJudge | None = None, *, max_hops: int = 3
    ) -> None:
        self._inner = inner
        self._judge = judge or ClaudeSufficiencyJudge()
        self._max_hops = max_hops
        self.last_hops: tuple[HopRecord, ...] = ()

    def retrieve(
        self, query: str, *, k: int = 10, filters: Filter | None = None
    ) -> list[Retrieved]:
        if k <= 0:
            self.last_hops = ()
            return []

        # chunk_id -> (Retrieved as first stamped, best score seen). Keeping
        # the *earliest* hop's stamp is the most meaningful answer to "which
        # hop found this" for a chunk re-found by a later hop, but the score
        # should never regress just because a later hop re-found it weaker.
        best_by_id: dict[str, Retrieved] = {}
        hops: list[HopRecord] = []
        current_query = query

        for hop_index in range(self._max_hops):
            hop_results = self._inner.retrieve(current_query, k=k, filters=filters)
            for r in hop_results:
                chunk_id = r.chunk.chunk_id
                if chunk_id not in best_by_id:
                    best_by_id[chunk_id] = _stamp_hop(r, hop_index, current_query)
                elif r.score > best_by_id[chunk_id].score:
                    existing = best_by_id[chunk_id]
                    best_by_id[chunk_id] = dataclasses.replace(existing, score=r.score)

            decision = self._judge.assess(query, hops, list(best_by_id.values()))
            hops.append(
                HopRecord(
                    hop_index=hop_index,
                    query=current_query,
                    retrieved_chunk_ids=tuple(r.chunk.chunk_id for r in hop_results),
                    sufficient=decision.sufficient,
                    reasoning=decision.reasoning,
                )
            )

            if decision.sufficient or not decision.next_query:
                break
            current_query = decision.next_query

        self.last_hops = tuple(hops)
        merged = sorted(best_by_id.values(), key=lambda r: r.score, reverse=True)
        return merged[:k]
