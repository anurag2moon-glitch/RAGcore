"""HybridRetriever — fuses a vector-similarity channel and a keyword (BM25)
channel via reciprocal rank fusion (RRF), before anything else in the stack
sees a result. A Retriever in its own right (a peer of VectorRetriever, not
a decorator around it — it owns both channels directly), and the base of the
default stack: GraphRetriever/ConceptRetriever/etc. wrap it exactly the way
they'd wrap a plain VectorRetriever.

RRF decides the candidate *set*; it never decides a result's final *score*.
Every downstream layer (GraphRetriever, ConceptRetriever, MMRReranker)
assumes one comparable score scale — LanceDBStore's own
`1 / (1 + distance)`, roughly 0.33-1.0 (see ragcore.retrieval.scoring).
Raw RRF scores are tiny (`1/61 ≈ 0.016` for rank 1 at the default rrf_k=60)
and on a completely different scale. If HybridRetriever emitted them
directly as `Retrieved.score`, every ConceptRetriever/GraphRetriever
expansion downstream (scored on the 0.33-1.0 scale) would systematically
outrank every base hit the moment they're merged into one list — silently
inverting the "expansions rank below seeds" invariant those layers depend
on. So: RRF fuses ranks from both channels to decide which chunks make the
candidate set (and, via `extra["rrf_score"]`/`extra["rrf_rank"]`, in what
preference order), then every winner is (re-)scored with the same cosine
transform every other layer uses, against its own stored vector.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Sequence

from ragcore.embedding.base import Embedder
from ragcore.lexical.base import LexicalStore
from ragcore.retrieval._filters import matches_filter
from ragcore.retrieval.scoring import cosine_score
from ragcore.stores.base import VectorStore
from ragcore.types import Chunk, Filter, Retrieved

if TYPE_CHECKING:
    from ragcore.corpus import Corpus


def _source_ids_from_filter(filters: Filter | None) -> Sequence[str] | None:
    """A source_id-only filter can be pushed down to the lexical store's own
    SQL as a fast path (see Fts5LexicalStore.search). Any other filter shape
    returns None, so HybridRetriever falls back to a Python post-filter
    (ragcore.retrieval._filters.matches_filter) applied after hydration."""
    if not filters or set(filters.keys()) != {"source_id"}:
        return None
    value = filters["source_id"]
    if isinstance(value, dict):
        return None  # a range/eq-operator dict, not plain equality/membership
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        lexical: LexicalStore,
        corpus: Corpus,
        *,
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        lexical_weight: float = 1.0,
        channel_multiplier: int = 2,
        max_per_channel: int = 200,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._lexical = lexical
        self._corpus = corpus
        self._rrf_k = rrf_k
        self._vector_weight = vector_weight
        self._lexical_weight = lexical_weight
        self._channel_multiplier = channel_multiplier
        self._max_per_channel = max_per_channel

    def retrieve(
        self, query: str, *, k: int = 10, filters: Filter | None = None
    ) -> list[Retrieved]:
        if k <= 0:
            return []

        fetch = min(max(k * self._channel_multiplier, 20), self._max_per_channel)

        query_vec = self._embedder.embed_query(query)
        vector_hits = self._store.query(query_vec, fetch, filters)

        source_ids = _source_ids_from_filter(filters)
        lexical_hits = self._lexical.search(query, fetch, source_ids=source_ids)

        rrf_scores: dict[str, float] = {}
        chunks_by_id: dict[str, Chunk] = {}
        for rank, r in enumerate(vector_hits, start=1):
            chunk_id = r.chunk.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + self._vector_weight / (
                self._rrf_k + rank
            )
            chunks_by_id[chunk_id] = r.chunk

        lexical_only_ids = [h.chunk_id for h in lexical_hits if h.chunk_id not in chunks_by_id]
        if lexical_only_ids:
            for chunk in self._corpus.get(lexical_only_ids):
                chunks_by_id[chunk.chunk_id] = chunk

        # A general (non-source_id) filter can't be evaluated by the lexical
        # store itself — apply it in Python, but only to lexical-only hits;
        # a hit that's also a vector hit already satisfied `filters` via the
        # store's own query.
        vector_hit_ids = {r.chunk.chunk_id for r in vector_hits}
        needs_post_filter = filters is not None and source_ids is None
        for h in lexical_hits:
            chunk = chunks_by_id.get(h.chunk_id)
            if chunk is None:
                continue  # hydration miss (e.g. deleted after indexing) — skip, don't crash
            is_lexical_only = h.chunk_id not in vector_hit_ids
            if is_lexical_only and needs_post_filter and not matches_filter(chunk.meta, filters):
                continue
            rrf_scores[h.chunk_id] = rrf_scores.get(h.chunk_id, 0.0) + self._lexical_weight / (
                self._rrf_k + h.rank
            )

        rrf_rank_by_id = {
            chunk_id: rank
            for rank, chunk_id in enumerate(
                sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True), start=1
            )
        }

        results: list[Retrieved] = []
        for chunk_id, chunk in chunks_by_id.items():
            if chunk_id not in rrf_scores:
                continue  # filtered out above
            score = cosine_score(query_vec, chunk.vector)
            extra = {
                **chunk.meta.extra,
                "rrf_score": rrf_scores[chunk_id],
                "rrf_rank": rrf_rank_by_id[chunk_id],
            }
            stamped = dataclasses.replace(chunk, meta=dataclasses.replace(chunk.meta, extra=extra))
            results.append(Retrieved(chunk=stamped, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
