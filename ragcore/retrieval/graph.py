"""GraphRetriever — decorates a vector Retriever with one-hop (configurable)
graph expansion. Implements the exact same Retriever protocol as
VectorRetriever, so wiring it in is a one-line swap:

    corpus.retriever = GraphRetriever(corpus.retriever, graph_store, corpus)

Every existing `corpus.retrieve(...)` call site — including `Corpus.ask()`,
which just calls `self.retrieve()` — is unaffected and becomes transparently
graph-aware. Vector search always runs first (seeds); graph traversal only
ever *adds* chunks the embedding might have missed, it never removes a
vector hit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragcore.graph.base import GraphStore
from ragcore.graph.types import GraphContext
from ragcore.retrieval.base import Retriever
from ragcore.types import Filter, Retrieved

if TYPE_CHECKING:
    # Deferred: this module is only ever duck-typed against `corpus`
    # (.get(...)), and importing ragcore.corpus at module level here risks
    # the same import-order cycle documented in ragcore/graph/pipeline.py.
    from ragcore.corpus import Corpus


class GraphRetriever:
    def __init__(
        self,
        inner: Retriever,
        graph: GraphStore,
        corpus: Corpus,
        *,
        hops: int = 1,
        expansion_budget: int = 30,
        expansion_score_decay: float = 0.5,
        graph_reserve: int | None = None,
    ) -> None:
        self._inner = inner
        self._graph = graph
        self._corpus = corpus
        self._hops = hops
        self._expansion_budget = expansion_budget
        self._decay = expansion_score_decay
        self._graph_reserve = graph_reserve

    def retrieve(
        self, query: str, *, k: int = 10, filters: Filter | None = None
    ) -> list[Retrieved]:
        if k <= 0:
            return []

        # Every expanded result is scored below every seed (see the decay
        # heuristic below), so if all k seed slots were filled from vector
        # search first, `merged[:k]` would always truncate every expansion
        # away — graph traversal becomes dead code the moment vector search
        # alone can fill k, which is the normal case once the corpus is
        # bigger than the query's own semantic neighborhood. Reserving a
        # slice of k up front makes expansion reachable in practice, not
        # just when the vector store happens to return fewer than k hits.
        # A single `inner.retrieve(k=k)` call still covers both: entity/
        # neighbor lookups run only against the first `seed_k` of them, and
        # the remainder becomes a same-call fallback pool (`reserve_pool`)
        # used to top back up to k if expansion contributes less than the
        # reserve (or nothing at all) — so a query with no graph signal
        # still gets a full k vector-only results, not k - reserve.
        reserve = self._graph_reserve if self._graph_reserve is not None else max(1, k // 3)
        reserve = max(0, min(reserve, k - 1))
        seed_k = k - reserve

        all_seeds = self._inner.retrieve(query, k=k, filters=filters)
        if not all_seeds:
            return all_seeds

        seeds = list(all_seeds[:seed_k])
        reserve_pool = all_seeds[seed_k:]

        expanded_results: list[Retrieved] = []
        seed_chunk_ids = [r.chunk.chunk_id for r in seeds]
        seed_entities_by_chunk = self._graph.entities_for_chunks(seed_chunk_ids)
        seed_entity_ids = {
            e.entity_id for entities in seed_entities_by_chunk.values() for e in entities
        }

        if seed_entity_ids:
            relations = self._graph.neighbors(list(seed_entity_ids), hops=self._hops)
            neighbor_entity_ids = {
                eid for rel in relations for eid in (rel.source_entity_id, rel.target_entity_id)
            } - seed_entity_ids

            if neighbor_entity_ids:
                chunks_by_entity = self._graph.chunks_for_entities(list(neighbor_entity_ids))
                seed_id_set = set(seed_chunk_ids)

                # Which entity(ies) pulled in each newly-discovered chunk,
                # for GraphContext provenance.
                new_chunk_entities: dict[str, list[str]] = {}
                for entity_id, chunk_ids in chunks_by_entity.items():
                    for cid in chunk_ids:
                        if cid in seed_id_set:
                            continue
                        new_chunk_entities.setdefault(cid, []).append(entity_id)

                new_chunk_ids = list(new_chunk_entities)[: self._expansion_budget]
                if new_chunk_ids:
                    expanded_chunks = self._corpus.get(new_chunk_ids)
                    # Simple decay heuristic, not a calibrated score:
                    # expanded results rank below every seed, ordered
                    # relative to each other only by this same constant.
                    # Good enough for v1 — see design doc addendum.
                    min_seed_score = min(r.score for r in seeds)
                    expanded_results = [
                        Retrieved(
                            chunk=c,
                            score=min_seed_score * self._decay,
                            graph=GraphContext(
                                entities=tuple(new_chunk_entities[c.chunk_id]),
                                path=tuple(seed_entity_ids) + tuple(new_chunk_entities[c.chunk_id]),
                            ),
                        )
                        for c in expanded_chunks
                    ]

        used_ids = {r.chunk.chunk_id for r in seeds} | {r.chunk.chunk_id for r in expanded_results}
        total = len(seeds) + len(expanded_results)
        if total < k:
            for r in reserve_pool:
                if r.chunk.chunk_id in used_ids:
                    continue
                seeds.append(r)
                used_ids.add(r.chunk.chunk_id)
                total += 1
                if total >= k:
                    break

        merged = seeds + expanded_results
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:k]
