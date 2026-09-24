"""HierarchyRetriever — decorates a Retriever with structural zoom-level
awareness: pulls in relevant section/chapter/book summaries (see
ragcore.hierarchy) as synthetic chunks, alongside the inner retriever's own
leaf-chunk hits, so retrieval isn't limited to chunk-level granularity.

Simpler than ConceptRetriever/ConceptMapper by design: hierarchy labels are
structural (chapter/section/book titles), not open-ended thematic
propositions, so there is no cheap-embedding-shortlist-vs-LLM-refinement
tradeoff worth a separate mapper Protocol here — matching is
embedding-similarity only, against every stored HierarchyNode's own
precomputed vector, scored with the same cosine transform
(ragcore.retrieval.scoring.cosine_score) every other layer uses.

Unlike a community-summary synthetic chunk (see ragcore.retrieval.concept),
a hierarchy-summary synthetic chunk carries the *real* book's source_id, not
a synthetic cross-source one — a chapter summary is a within-source
abstraction that should compete with that same book's own literal chunks
for context budget, not be exempt from it (see
ragcore.retrieval.mmr.MMRReranker's per_source_exempt_kinds).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ragcore.chunking.tokenizer import count_tokens
from ragcore.hierarchy.base import HierarchyStore
from ragcore.hierarchy.types import HierarchyLevel, HierarchyNode
from ragcore.retrieval.base import Retriever
from ragcore.retrieval.scoring import cosine_score
from ragcore.types import Chunk, ChunkMeta, Filter, Retrieved

if TYPE_CHECKING:
    from ragcore.embedding.base import Embedder


def _node_chunk(node: HierarchyNode) -> Chunk:
    """Represents a HierarchyNode as a synthetic Chunk so it flows through
    the rest of retrieval (MMR, the generator, verification) unchanged —
    see ChunkOrigin's docstring in ragcore.types and CLAUDE.md's
    synthetic-chunk convention."""
    text = node.summary
    meta = ChunkMeta(
        source_id=node.source_id,
        source_path=node.source_id,
        source_title=f"{node.level.capitalize()} summary: {node.label}",
        source_type="synthetic",
        section_path=(),
        chapter_title=node.label,
        chapter_index=None,
        section=None,
        page_start=None,
        page_end=None,
        chunk_index=0,
        context_group_id=node.node_id,
        char_start=0,
        char_end=len(text),
        token_count=count_tokens(text),
        extra={
            "ragcore_kind": "hierarchy_summary",
            "hierarchy_level": node.level,
            "node_id": node.node_id,
        },
    )
    return Chunk(
        chunk_id=node.node_id,
        text=text,
        meta=meta,
        vector=list(node.vector) if node.vector is not None else None,
    )


class HierarchyRetriever:
    """Retriever decorator combining the inner retriever's own hits with
    relevant section/chapter/book summaries, into one ranked list. Reuses
    ragcore.retrieval.graph.GraphRetriever's proven reservation scheme."""

    def __init__(
        self,
        inner: Retriever,
        hierarchy: HierarchyStore,
        embedder: Embedder,
        *,
        node_reserve: int | None = None,
        max_nodes: int = 3,
        levels: Sequence[HierarchyLevel] = ("section", "chapter", "book"),
    ) -> None:
        self._inner = inner
        self._hierarchy = hierarchy
        self._embedder = embedder
        self._node_reserve = node_reserve
        self._max_nodes = max_nodes
        self._levels = tuple(levels)

    def retrieve(
        self, query: str, *, k: int = 10, filters: Filter | None = None
    ) -> list[Retrieved]:
        if k <= 0:
            return []

        reserve = self._node_reserve if self._node_reserve is not None else max(1, k // 4)
        reserve = max(0, min(reserve, k - 1))
        seed_k = k - reserve

        all_seeds = self._inner.retrieve(query, k=k, filters=filters)
        if not all_seeds:
            return all_seeds

        seeds = list(all_seeds[:seed_k])
        reserve_pool = all_seeds[seed_k:]
        seed_id_set = {r.chunk.chunk_id for r in seeds}

        candidates = [
            n
            for level in self._levels
            for n in self._hierarchy.nodes_by_level(level)
            if n.vector is not None
        ]

        node_results: list[Retrieved] = []
        if candidates:
            query_vec = self._embedder.embed_query(query)
            scored = sorted(
                ((n, cosine_score(query_vec, n.vector)) for n in candidates),
                key=lambda pair: pair[1],
                reverse=True,
            )[: self._max_nodes]
            for node, score in scored:
                if node.node_id in seed_id_set:
                    continue
                node_results.append(Retrieved(chunk=_node_chunk(node), score=score))

        used_ids = seed_id_set | {r.chunk.chunk_id for r in node_results}
        total = len(seeds) + len(node_results)
        if total < k:
            for r in reserve_pool:
                if r.chunk.chunk_id in used_ids:
                    continue
                seeds.append(r)
                used_ids.add(r.chunk.chunk_id)
                total += 1
                if total >= k:
                    break

        merged = seeds + node_results
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:k]
