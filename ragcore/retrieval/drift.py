"""DriftRetriever and graph_search_modes — a named local/global/DRIFT
search-mode selector composing the existing GraphRetriever/ConceptRetriever
layers explicitly (docs/ARCHITECTURE.md Phase 4 gap list item 20), as an
alternative to always stacking them unconditionally the way
ragcore.retrieval.stack.default_stack() does.

Naming matches ragcore.retrieval.routing's existing mode vocabulary
("graph_local"/"graph_global") exactly, so `graph_search_modes()`'s output
plugs directly into `RoutedRetriever` with no renaming:

    modes = graph_search_modes(base, corpus, graph_store=g, concepts_store=c,
                                embedder=e)
    retriever = RoutedRetriever(modes, ClaudeQueryRouter())

`DriftRetriever` is a *fused-single-call* approximation of DRIFT (Dynamic
Reasoning and Inference with Flexible Traversal, from Microsoft's GraphRAG
paper), stated plainly rather than overclaimed: the paper's DRIFT primes
with a global community-summary answer, decomposes it into targeted local
follow-up queries, and iteratively refines — a multi-step, query-dependent
process. This implementation instead retrieves via the local
(GraphRetriever, entity-neighborhood) and global (ConceptRetriever,
concept/community-summary) paths for the *same* query in one call, fusing
both result sets via `ragcore.retrieval.fusion.rrf_merge` — the same
reasoning every other RRF fusion in this codebase states: a community-
summary hit's score isn't on a scale directly comparable to an entity-
expansion hit's score, so rank (not raw score) is what's fused on. This
gets DRIFT's core *value* (broad thematic grounding plus specific entity-
level detail, in one answer) without its iterative query-decomposition
machinery — a real, useful simplification, not a mislabeled reimplementation
of the paper's actual algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragcore.retrieval.base import Retriever
from ragcore.retrieval.concept import ConceptRetriever
from ragcore.retrieval.fusion import rrf_merge
from ragcore.retrieval.graph import GraphRetriever
from ragcore.types import Filter, Retrieved

if TYPE_CHECKING:
    from ragcore.concepts.base import ConceptMapper, ConceptStore
    from ragcore.corpus import Corpus
    from ragcore.embedding.base import Embedder
    from ragcore.graph.base import GraphStore


class DriftRetriever:
    """Retrieves via a local (entity-neighborhood) and a global
    (concept/community-summary) Retriever for the same query, fusing both
    via RRF. See module docstring for the full DRIFT-naming rationale."""

    def __init__(self, local: Retriever, global_: Retriever, *, rrf_k: int = 60) -> None:
        self._local = local
        self._global = global_
        self._rrf_k = rrf_k

    def retrieve(
        self, query: str, *, k: int = 10, filters: Filter | None = None
    ) -> list[Retrieved]:
        if k <= 0:
            return []
        local_hits = self._local.retrieve(query, k=k, filters=filters)
        global_hits = self._global.retrieve(query, k=k, filters=filters)
        return rrf_merge(
            [("local", local_hits), ("global", global_hits)],
            k=k,
            rrf_k=self._rrf_k,
            extra_key="drift_sources",
        )


def graph_search_modes(
    base: Retriever,
    corpus: Corpus,
    *,
    graph_store: GraphStore,
    concepts_store: ConceptStore,
    embedder: Embedder,
    concept_mapper: ConceptMapper | None = None,
    hops: int = 1,
    rrf_k: int = 60,
) -> dict[str, Retriever]:
    """Builds `{"graph_local": ..., "graph_global": ..., "drift": ...}`,
    ready to hand to `RoutedRetriever` alongside `base` itself (register
    `base` under `"vector"`/`"keyword"` separately — this helper only
    covers the graph-aware modes, since it needs `graph_store`/
    `concepts_store` neither the plain base retriever nor RoutedRetriever
    itself requires). Each mode is built independently from the same
    `base`, not chained — "local" and "global" each add their own
    expansion directly on top of the shared base search, rather than one
    stacking on the other's already-expanded result."""
    local = GraphRetriever(base, graph_store, corpus, hops=hops)
    global_ = ConceptRetriever(base, concepts_store, corpus, embedder, concept_mapper)
    drift = DriftRetriever(local, global_, rrf_k=rrf_k)
    return {"graph_local": local, "graph_global": global_, "drift": drift}
