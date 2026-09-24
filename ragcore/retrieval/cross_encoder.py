"""CrossEncoderReranker — decorates a Retriever with a cross-encoder-style
reranking pass: scores (query, chunk_text) pairs directly via a Reranker
backend, for the top-N candidates coming out of retrieval, before diversity
reranking.

Ordering: this runs BEFORE MMRReranker, not after or alongside it.
MMRReranker's job is to diversify among an *already relevance-ranked*
candidate set; running MMR first (on the coarser embedding/RRF-derived
scores) and cross-encoder reranking after would let the more accurate
signal influence only *display order* within whatever set MMR already
committed to, never *which* documents made that set — defeating the point
of using a more accurate signal at all. See ragcore/retrieval/stack.py for
the full default composition.

Does not overwrite `Retrieved.score` — `Corpus.retrieve(..., min_score=...)`
compares against the original embedding-derived scale, and silently
redefining that scale here would change what a caller's threshold means
without them knowing. The cross-encoder's score is instead stamped into
`chunk.meta.extra["cross_encoder_score"]`; `MMRReranker` (see
ragcore/retrieval/mmr.py) prefers it over `.score` when present, but
all-or-nothing per query — never mixing two different scales inside one
min-max normalization.
"""

from __future__ import annotations

import dataclasses

from ragcore.reranking.base import Reranker
from ragcore.retrieval.base import Retriever
from ragcore.types import Filter, Retrieved


class CrossEncoderReranker:
    """`fetch_multiplier=1` (the default) is correct when this sits directly
    under MMRReranker in the default stack — MMR already over-fetches 4x on
    its own side, so multiplying again here would send far more documents
    to the reranking backend per query for no benefit. Set
    `fetch_multiplier=3` (or similar) if this is wired as the outermost
    layer instead, with nothing downstream doing its own over-fetch.
    """

    def __init__(
        self,
        inner: Retriever,
        reranker: Reranker,
        *,
        fetch_multiplier: int = 1,
        max_candidates: int = 100,
    ) -> None:
        self._inner = inner
        self._reranker = reranker
        self._fetch_multiplier = fetch_multiplier
        self._max_candidates = max_candidates

    def retrieve(
        self, query: str, *, k: int = 10, filters: Filter | None = None
    ) -> list[Retrieved]:
        if k <= 0:
            return []

        fetch_k = min(max(k * self._fetch_multiplier, k), self._max_candidates)
        pool = self._inner.retrieve(query, k=fetch_k, filters=filters)
        if not pool:
            return pool

        # Score every candidate — top_n=None — so this decorator itself
        # decides what to keep, rather than trusting the backend's own
        # notion of "top N" to line up with our `k`.
        hits = self._reranker.rerank(query, [r.chunk.text for r in pool], top_n=None)

        rescored: list[Retrieved] = []
        for hit in hits:
            r = pool[hit.index]
            extra = {**r.chunk.meta.extra, "cross_encoder_score": hit.relevance_score}
            stamped_chunk = dataclasses.replace(r.chunk, meta=dataclasses.replace(r.chunk.meta, extra=extra))
            rescored.append(dataclasses.replace(r, chunk=stamped_chunk))

        rescored.sort(key=lambda r: r.chunk.meta.extra["cross_encoder_score"], reverse=True)
        return rescored[:k]
