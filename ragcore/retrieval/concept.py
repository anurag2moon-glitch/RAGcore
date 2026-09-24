"""ConceptRetriever — decorates a Retriever with concept-graph-aware
expansion: chunks linked to query-relevant concepts, plus relevant
precomputed community summaries, folded into the same ranked result list as
the inner retriever's own hits.

Reuses ragcore.retrieval.graph.GraphRetriever's proven reservation scheme
rather than inventing a new one: reserve a fraction of `k` up front, keep the
unused remainder of the inner call as a backfill pool, and only actually pay
for expansion when there's a real signal for it — a query with no relevant
concepts still returns a full `k` of the inner retriever's own hits. See
GraphRetriever's docstring for the failure mode this avoids (expanded
results scoring below every seed get truncated away entirely unless a slice
is reserved for them).

Concept-linked chunks and community summaries are scored with a real
query-vector cosine under the identical `1 / (1 + distance)` transform
ragcore.stores.lancedb.LanceDBStore.query() uses (`distance = 1 - cosine`),
not an invented constant — this is what lets a mixed pool of vector hits,
graph expansions, concept expansions, and summaries sort correctly against
each other by simple descending score, which is also what naturally
deprioritizes GraphRetriever's already-lower-scored expansions (see its
`expansion_score_decay`) relative to concept hits without any special-case
eviction logic here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragcore.chunking.tokenizer import count_tokens
from ragcore.concepts.base import ConceptMapper, ConceptStore
from ragcore.concepts.mapper import EmbeddingConceptMapper
from ragcore.retrieval.base import Retriever
from ragcore.retrieval.scoring import cosine_score
from ragcore.types import Chunk, ChunkMeta, Filter, Retrieved

if TYPE_CHECKING:
    from ragcore.corpus import Corpus
    from ragcore.embedding.base import Embedder


def _summary_chunk(summary) -> Chunk:
    """Represents a CommunitySummary as a synthetic Chunk so it flows
    through the rest of retrieval (MMR, the generator, verification)
    unchanged — see ChunkOrigin's docstring in ragcore.types and CLAUDE.md's
    synthetic-chunk convention."""
    # summary.community_id is already namespaced (ragcore.concepts.communities
    # generates ids as f"community:{digest}") — used as-is, never re-prefixed.
    text = summary.summary
    meta = ChunkMeta(
        source_id=summary.community_id,
        source_path=summary.community_id,
        source_title=f"Community summary: {summary.label}",
        source_type="synthetic",
        section_path=(),
        chapter_title=summary.label,
        chapter_index=None,
        section=None,
        page_start=None,
        page_end=None,
        chunk_index=0,
        context_group_id=summary.community_id,
        char_start=0,
        char_end=len(text),
        token_count=count_tokens(text),
        extra={
            "ragcore_kind": "community_summary",
            "concept_ids": list(summary.concept_ids),
            "contributing_source_ids": list(summary.source_ids),
        },
    )
    return Chunk(
        chunk_id=summary.community_id,
        text=text,
        meta=meta,
        vector=list(summary.vector) if summary.vector is not None else None,
    )


class ConceptRetriever:
    """Retriever decorator combining (a) the inner retriever's own hits, (b)
    chunks linked to query-relevant concepts, and (c) relevant community
    summaries, into one ranked list.

    `mapper` defaults to EmbeddingConceptMapper(embedder) — no LLM call, so
    wiring ConceptRetriever with defaults adds zero extra LLM calls to
    Corpus.ask(). Pass a ClaudeConceptMapper (wrapping an
    EmbeddingConceptMapper) for sharper mapping at the cost of one call per
    query. `shortlist_size` is only used to build that default mapper; a
    caller-supplied `mapper` controls its own shortlist.
    """

    def __init__(
        self,
        inner: Retriever,
        concepts: ConceptStore,
        corpus: Corpus,
        embedder: Embedder,
        mapper: ConceptMapper | None = None,
        *,
        concept_reserve: int | None = None,
        summary_slots: int = 2,
        expansion_budget: int = 30,
        max_concepts: int = 8,
        shortlist_size: int = 60,
    ) -> None:
        self._inner = inner
        self._concepts = concepts
        self._corpus = corpus
        self._embedder = embedder
        self._mapper = mapper or EmbeddingConceptMapper(embedder, shortlist_size=shortlist_size)
        self._concept_reserve = concept_reserve
        self._summary_slots = summary_slots
        self._expansion_budget = expansion_budget
        self._max_concepts = max_concepts

    def retrieve(
        self, query: str, *, k: int = 10, filters: Filter | None = None
    ) -> list[Retrieved]:
        if k <= 0:
            return []

        # Deliberately below GraphRetriever's default k // 3 (see module
        # docstring): concept hits shouldn't crowd out graph expansions when
        # ConceptRetriever sits on top of one.
        reserve = self._concept_reserve if self._concept_reserve is not None else max(1, k // 4)
        reserve = max(0, min(reserve, k - 1))
        seed_k = k - reserve

        all_seeds = self._inner.retrieve(query, k=k, filters=filters)
        if not all_seeds:
            return all_seeds

        seeds = list(all_seeds[:seed_k])
        reserve_pool = all_seeds[seed_k:]
        seed_id_set = {r.chunk.chunk_id for r in seeds}

        # store.all_concepts() reads the whole concept graph once per query
        # — fine at the concept-graph scale this layer targets (far smaller
        # than the chunk/entity counts elsewhere in the pipeline); revisit
        # with an index if a corpus's concept vocabulary grows very large.
        all_concepts = self._concepts.all_concepts()
        mapped_ids = self._mapper.map(query, all_concepts)[: self._max_concepts]

        expanded_results: list[Retrieved] = []
        summary_results: list[Retrieved] = []

        if mapped_ids:
            query_vec = self._embedder.embed_query(query)

            chunks_by_concept = self._concepts.chunks_for_concepts(mapped_ids)
            new_chunk_ids = list(
                dict.fromkeys(
                    cid
                    for ids in chunks_by_concept.values()
                    for cid in ids
                    if cid not in seed_id_set
                )
            )[: self._expansion_budget]

            if new_chunk_ids:
                concepts_by_chunk: dict[str, list[str]] = {}
                for concept_id, chunk_ids in chunks_by_concept.items():
                    for cid in chunk_ids:
                        concepts_by_chunk.setdefault(cid, []).append(concept_id)

                for chunk in self._corpus.get(new_chunk_ids):
                    expanded_results.append(
                        Retrieved(chunk=chunk, score=cosine_score(query_vec, chunk.vector))
                    )

            summaries = self._concepts.communities_for_concepts(mapped_ids)[: self._summary_slots]
            for summary in summaries:
                chunk = _summary_chunk(summary)
                summary_results.append(
                    Retrieved(chunk=chunk, score=cosine_score(query_vec, summary.vector))
                )

        used_ids = (
            seed_id_set
            | {r.chunk.chunk_id for r in expanded_results}
            | {r.chunk.chunk_id for r in summary_results}
        )
        total = len(seeds) + len(expanded_results) + len(summary_results)
        if total < k:
            for r in reserve_pool:
                if r.chunk.chunk_id in used_ids:
                    continue
                seeds.append(r)
                used_ids.add(r.chunk.chunk_id)
                total += 1
                if total >= k:
                    break

        merged = seeds + expanded_results + summary_results
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:k]
