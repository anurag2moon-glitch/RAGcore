"""MMRReranker — decorates a Retriever with maximal-marginal-relevance
diversity reranking and a per-source-document cap, so the final context
handed to generation isn't dominated by several chunks from the same
chapter or book.

Needs no Embedder of its own: query-document relevance reuses whatever
`score` the inner retriever already assigned (LanceDB's, GraphRetriever's
decayed expansion score, ConceptRetriever's cosine-derived score, all now on
one comparable scale — see ragcore.retrieval.concept's module docstring) —
or, when every candidate in the pool carries a
`chunk.meta.extra["cross_encoder_score"]` (see
ragcore.retrieval.cross_encoder.CrossEncoderReranker), that instead, since
it's the more accurate signal. That switch is all-or-nothing per query:
mixing two different score scales inside one min-max normalization is worse
than consistently using the weaker one. Document-document similarity reuses
Chunk.vector, populated on every retrieval path that matters (see
ragcore.stores.lancedb.LanceDBStore.query/get/get_by_source, all sharing
`_row_to_chunk`).

Outermost in the stack (below CrossEncoderReranker's ordering rule, see its
module docstring): MMRReranker(CrossEncoderReranker(...(HybridRetriever(...)))).
Over-fetching from `inner` is safe against every inner decorator's own
fractional reservation scheme — see GraphRetriever and ConceptRetriever's
docstrings; asking for `k * fetch_multiplier` scales each layer's reserve
proportionally rather than compounding, since every decorator here calls its
own inner with the *full* k it was given, not a shrunk one.

**Per-kind summary caps.** A community summary (see
ragcore.retrieval.concept) is cross-source synthesis with no single owning
book — its synthetic `source_id` is exempt from the per-source cap
entirely. A hierarchy summary (see ragcore.retrieval.hierarchy) is a
within-source abstraction that directly competes with that same book's own
literal chunks for context budget — it carries the real book's `source_id`
and is *not* exempt from the per-source cap, on top of having its own,
separate, non-relaxing `summary_caps["hierarchy_summary"]` budget. One
shared cap across both kinds would let a chapter summary evict a community
summary (or vice versa) for reasons unrelated to either's value — hence two
independent caps, keyed by `chunk.meta.extra["ragcore_kind"]`.
`max_summaries` is kept as a backward-compatible alias that seeds
`summary_caps["community_summary"]` when `summary_caps` itself isn't given.
"""

from __future__ import annotations

import re
from typing import Literal

from ragcore.errors import RagcoreError
from ragcore.graph.resolver import _cosine
from ragcore.retrieval.base import Retriever
from ragcore.types import Filter, Retrieved

_WORD_RE = re.compile(r"[a-z0-9]+")

_DEFAULT_SUMMARY_CAPS = {"community_summary": 2, "hierarchy_summary": 2}
_DEFAULT_PER_SOURCE_EXEMPT_KINDS = frozenset({"community_summary"})


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _kind_of(r: Retrieved) -> str | None:
    return r.chunk.meta.extra.get("ragcore_kind")


def _similarity(a: Retrieved, b: Retrieved, *, on_missing_vectors: str) -> float:
    if a.chunk.vector is not None and b.chunk.vector is not None:
        return _cosine(a.chunk.vector, b.chunk.vector)
    if on_missing_vectors == "lexical":
        return _jaccard(a.chunk.text, b.chunk.text)
    return 0.0  # unreachable for "passthrough"/"error" — those bail out before the loop runs


class MMRReranker:
    """Retriever decorator: over-fetches from `inner`, then greedily selects
    `k` results balancing relevance against diversity from what's already
    been picked, honoring a per-source cap that softens (never hardens) —
    diversity is traded against ordering, never against the inner
    retriever's own recall.
    """

    def __init__(
        self,
        inner: Retriever,
        *,
        lambda_: float = 0.7,
        fetch_multiplier: int = 4,
        max_candidates: int = 100,
        per_source_cap: int | None = 3,
        max_summaries: int | None = None,
        summary_caps: dict[str, int] | None = None,
        per_source_exempt_kinds: frozenset[str] = _DEFAULT_PER_SOURCE_EXEMPT_KINDS,
        on_missing_vectors: Literal["passthrough", "lexical", "error"] = "passthrough",
    ) -> None:
        self._inner = inner
        self._lambda = lambda_
        self._fetch_multiplier = fetch_multiplier
        self._max_candidates = max_candidates
        self._per_source_cap = per_source_cap
        if summary_caps is not None:
            self._summary_caps = dict(summary_caps)
        elif max_summaries is not None:
            self._summary_caps = {**_DEFAULT_SUMMARY_CAPS, "community_summary": max_summaries}
        else:
            self._summary_caps = dict(_DEFAULT_SUMMARY_CAPS)
        self._per_source_exempt_kinds = frozenset(per_source_exempt_kinds)
        self._on_missing_vectors = on_missing_vectors

    def retrieve(
        self, query: str, *, k: int = 10, filters: Filter | None = None
    ) -> list[Retrieved]:
        if k <= 0:
            return []

        fetch_k = min(k * self._fetch_multiplier, self._max_candidates)
        pool = self._inner.retrieve(query, k=fetch_k, filters=filters)

        # Nothing to diversify — a no-op on a small corpus rather than a
        # silent degradation.
        if len(pool) <= k:
            return pool

        if any(r.chunk.vector is None for r in pool):
            if self._on_missing_vectors == "passthrough":
                return pool[:k]
            if self._on_missing_vectors == "error":
                raise RagcoreError(
                    "MMRReranker: a candidate has no vector and on_missing_vectors='error'; "
                    "use 'lexical' for a token-overlap fallback or 'passthrough' to skip reranking"
                )
            # "lexical": fall through — _similarity() falls back to Jaccard
            # per-pair as needed, so every candidate stays eligible.

        # Prefer the cross-encoder's relevance score when every candidate in
        # the pool has one — all-or-nothing, decided once per query (see
        # module docstring): mixing two different scales inside one min-max
        # normalization would be worse than consistently using the weaker
        # signal.
        use_cross_encoder = bool(pool) and all(
            "cross_encoder_score" in r.chunk.meta.extra for r in pool
        )

        def relevance_of(r: Retrieved) -> float:
            return r.chunk.meta.extra["cross_encoder_score"] if use_cross_encoder else r.score

        scores = [relevance_of(r) for r in pool]
        lo, hi = min(scores), max(scores)
        span = hi - lo
        norm = {
            r.chunk.chunk_id: ((relevance_of(r) - lo) / span if span > 0 else 1.0) for r in pool
        }

        selected: list[Retrieved] = []
        remaining = list(pool)
        source_counts: dict[str, int] = {}
        summary_counts: dict[str, int] = {}
        cap_disabled = False

        def eligible(r: Retrieved) -> bool:
            kind = _kind_of(r)
            if kind in self._summary_caps and summary_counts.get(kind, 0) >= self._summary_caps[kind]:
                return False
            if kind in self._per_source_exempt_kinds:
                return True
            if cap_disabled or self._per_source_cap is None:
                return True
            return source_counts.get(r.chunk.meta.source_id, 0) < self._per_source_cap

        def mmr_score(r: Retrieved) -> float:
            relevance = norm[r.chunk.chunk_id]
            if not selected:
                return relevance
            max_sim = max(
                _similarity(r, s, on_missing_vectors=self._on_missing_vectors) for s in selected
            )
            return self._lambda * relevance - (1 - self._lambda) * max_sim

        while remaining and len(selected) < k:
            candidates = [r for r in remaining if eligible(r)]
            if not candidates:
                if cap_disabled:
                    break  # nothing left eligible even with the cap relaxed (e.g. max_summaries exhausted)
                # Relax the per-source cap for the rest of this query rather
                # than ever returning fewer results than `inner` would have.
                cap_disabled = True
                continue

            best = max(candidates, key=mmr_score)
            selected.append(best)
            remaining.remove(best)
            kind = _kind_of(best)
            if kind in self._summary_caps:
                summary_counts[kind] = summary_counts.get(kind, 0) + 1
            if kind not in self._per_source_exempt_kinds:
                sid = best.chunk.meta.source_id
                source_counts[sid] = source_counts.get(sid, 0) + 1

        return selected
