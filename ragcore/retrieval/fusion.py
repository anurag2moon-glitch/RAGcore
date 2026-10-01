"""rrf_merge — shared reciprocal rank fusion helper for any retrieval
layer that issues more than one underlying retrieve() call per query and
needs to merge the results into one ranked, deduplicated list without
assuming the different calls' raw scores are on a comparable scale.

Extracted from ragcore.retrieval.query_expansion.QueryExpansionRetriever
(its first user) once ragcore.retrieval.routing.RoutedRetriever needed
the identical merge — same reasoning both already state independently:
a HyDE passage embeds in a different region of vector space than a short
question (QueryExpansionRetriever's case), and a keyword-mode result's
BM25-derived score isn't on the same scale as a vector-mode result's
cosine score (RoutedRetriever's case) — in both cases, comparing *rank*
within each list is meaningful where comparing raw *score* across lists
isn't. This mirrors ragcore.retrieval.hybrid.HybridRetriever's original
two-channel RRF for the same reason, generalized to any number of named
result lists.

RRF decides the candidate set and final order; `Retrieved.score` on each
result stays its own best real score from whichever contributing list
found it, never a raw RRF score — preserving the score-scale invariant
every layer in this stack depends on (see HybridRetriever's module
docstring for the fullest statement of this invariant).
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

from ragcore.types import Retrieved


def rrf_merge(
    labeled_results: Sequence[tuple[str, list[Retrieved]]],
    *,
    k: int,
    rrf_k: int = 60,
    extra_key: str = "rrf_sources",
) -> list[Retrieved]:
    """`labeled_results`: [(label, that call's own Retrieved list), ...].
    `extra_key`: where the tuple of contributing labels is stamped on each
    result's `ChunkMeta.extra`, for provenance/debugging."""
    rrf_scores: dict[str, float] = {}
    best_by_id: dict[str, Retrieved] = {}
    contributing: dict[str, list[str]] = {}

    for label, results in labeled_results:
        for rank, r in enumerate(results, start=1):
            chunk_id = r.chunk.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            contributing.setdefault(chunk_id, []).append(label)
            if chunk_id not in best_by_id or r.score > best_by_id[chunk_id].score:
                best_by_id[chunk_id] = r

    ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    out: list[Retrieved] = []
    for chunk_id in ranked_ids[:k]:
        r = best_by_id[chunk_id]
        extra = {**r.chunk.meta.extra, extra_key: tuple(contributing[chunk_id])}
        stamped = dataclasses.replace(r.chunk, meta=dataclasses.replace(r.chunk.meta, extra=extra))
        out.append(Retrieved(chunk=stamped, score=r.score))
    return out
