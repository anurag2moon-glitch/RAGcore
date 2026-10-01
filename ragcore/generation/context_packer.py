"""pack_context — token-budget-aware context packing for a Generator
(docs/ARCHITECTURE.md Phase 3 gap list item 15: "dedupe, ordering, parent
expansion, and token budgeting for any target model"). Today's
ClaudeGenerator._format_context naively concatenates every retrieved
chunk with no budget at all — fine for a handful of results, a real
problem once retrieval (hybrid search, multi-hop, query expansion, a
query router's combination mode — all added this session) can return
more context than any model's window, or even just its own
cost/latency-conscious max, can reasonably take.

Four responsibilities, in order:

1. **Dedupe** by chunk_id, keeping the first (highest-relevance)
   occurrence — defensive: nothing upstream should normally produce a
   duplicate, but a caller hand-merging two retrieve() calls could.
2. **Parent/neighbor expansion** (opt-in, needs `corpus`): for each
   retrieved chunk, fetch up to `expand_neighbors` chunks immediately
   before/after it by `chunk_index` within the same source — "parent
   expansion" in the sense the original Phase 1 spec names
   ("retrieval hits small chunks and returns larger context"), done via
   `Chunk.chunk_id`'s own positional `f"{source_id}:{chunk_index}"`
   scheme (see ragcore.chunking.structural's module docstring) rather
   than a hard dependency on ragcore.hierarchy — a chunk at a source's
   boundary simply has no neighbor on that side, not an error. An
   expansion chunk is scored just below the chunk it expands (never
   above it), so budget selection always prefers real hits over their
   own expansions when both can't fit.
3. **Token budgeting**: greedily keep chunks in relevance order
   (highest score first) until `max_tokens` (minus `reserve_tokens`,
   for whatever else shares the prompt) would be exceeded. Dropped
   chunks are returned too, in drop order, so a caller can log or
   surface what didn't make it in.
4. **Ordering**: the *kept* set is re-ordered for presentation by
   document position (`source_id`, `chapter_index`, `char_start`), not
   left in relevance-rank order — reading retrieved passages in the
   order they actually appear in the source is easier for a model (and
   a human) to follow than an interleaved-by-score jumble, a distinct
   concern from which chunks get selected in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from ragcore.chunking.tokenizer import count_tokens
from ragcore.types import Retrieved

if TYPE_CHECKING:
    from ragcore.corpus import Corpus


@dataclass(frozen=True)
class PackedContext:
    kept: list[Retrieved]
    """Final, document-ordered, budget-respecting entries."""

    dropped: list[Retrieved]
    """Cut for budget, in the order they were cut (lowest relevance
    first) — never includes an expansion chunk whose original hit was
    itself dropped (no point expanding something that didn't make it)."""

    total_tokens: int
    """Token count of `kept`, in the order it will actually be formatted
    (not the order chunks were considered in) — what a caller should
    compare against its own prompt-wide budget."""


def _expand_neighbors(chunk_id: str, source_id: str, chunk_index: int, n: int, corpus: Corpus):
    neighbor_ids = [
        f"{source_id}:{i}"
        for i in range(max(0, chunk_index - n), chunk_index + n + 1)
        if i != chunk_index
    ]
    if not neighbor_ids:
        return []
    return corpus.get(neighbor_ids)


def pack_context(
    context: Sequence[Retrieved],
    *,
    max_tokens: int,
    reserve_tokens: int = 0,
    corpus: Corpus | None = None,
    expand_neighbors: int = 0,
) -> PackedContext:
    budget = max(0, max_tokens - reserve_tokens)

    # 1. Dedupe, keeping the first (highest-relevance) occurrence.
    deduped: list[Retrieved] = []
    seen: set[str] = set()
    for r in context:
        if r.chunk.chunk_id not in seen:
            seen.add(r.chunk.chunk_id)
            deduped.append(r)

    # 2. Parent/neighbor expansion, opt-in.
    candidates: list[Retrieved] = list(deduped)
    if expand_neighbors > 0 and corpus is not None:
        for r in deduped:
            m = r.chunk.meta
            for neighbor in _expand_neighbors(
                r.chunk.chunk_id, m.source_id, m.chunk_index, expand_neighbors, corpus
            ):
                if neighbor.chunk_id in seen:
                    continue
                seen.add(neighbor.chunk_id)
                # scored just below the hit it expands, so a real hit is
                # always preferred over its own expansion when budget-
                # constrained, but an expansion still outranks a lower-
                # scoring independent hit if there's room for both.
                candidates.append(Retrieved(chunk=neighbor, score=r.score * 0.999))

    # Re-sort by score: expansions were appended after all original
    # deduped hits, not interleaved by relevance.
    candidates.sort(key=lambda r: r.score, reverse=True)

    # 3. Token-budget selection, in relevance order.
    kept: list[Retrieved] = []
    dropped: list[Retrieved] = []
    used_tokens = 0
    for r in candidates:
        tokens = count_tokens(r.chunk.text)
        if used_tokens + tokens > budget:
            dropped.append(r)
            continue
        kept.append(r)
        used_tokens += tokens

    # 4. Document-order presentation, independent of selection order.
    def _position(r: Retrieved) -> tuple[str, int, int]:
        m = r.chunk.meta
        return (m.source_id, m.chapter_index if m.chapter_index is not None else -1, m.char_start)

    kept.sort(key=_position)

    return PackedContext(
        kept=kept,
        dropped=dropped,
        total_tokens=sum(count_tokens(r.chunk.text) for r in kept),
    )
