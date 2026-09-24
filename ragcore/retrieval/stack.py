"""default_stack — a convenience builder returning the fully-wired default
retriever stack, so its composition order and tuned constants live in one
documented place rather than being re-derived by every consumer.

Order (outermost first), and why:

    MMRReranker(
      CrossEncoderReranker(
        IterativeRetriever(
          HierarchyRetriever(
            ConceptRetriever(
              GraphRetriever(
                HybridRetriever(embedder, store, lexical, corpus))))))))

- HybridRetriever is the base: fuses vector similarity and BM25 keyword
  search via reciprocal rank fusion (see its module docstring for why RRF
  decides the candidate *set* while a cosine transform decides the *score*).
- GraphRetriever, ConceptRetriever, and HierarchyRetriever each add their
  own expansion layer on top, in that order — entity-graph expansion first,
  concept/theme expansion (including community summaries) second, structural
  zoom-level expansion (section/chapter/book summaries) third.
- IterativeRetriever wraps the whole per-query stack, looping it with
  reformulated sub-queries when a query needs more than one hop.
- CrossEncoderReranker runs once over the final multi-hop union, *before*
  diversity reranking — see its module docstring for why that order (not
  the reverse) is the correct one.
- MMRReranker is outermost, diversifying the now-best-ranked candidate set.

Every layer is optional here: only the layers whose backing store you
actually provide are included, so an unfinished pipeline still composes
correctly — passing nothing but a corpus builds a plain
`MMRReranker(IterativeRetriever(VectorRetriever(...)))`... and even
`IterativeRetriever` is skipped unless `iterative=True` or a `judge` is
given (see its own note below).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragcore.retrieval.base import Retriever
from ragcore.retrieval.cross_encoder import CrossEncoderReranker
from ragcore.retrieval.hybrid import HybridRetriever
from ragcore.retrieval.iterative import IterativeRetriever, SufficiencyJudge
from ragcore.retrieval.mmr import MMRReranker
from ragcore.retrieval.vector import VectorRetriever

if TYPE_CHECKING:
    from ragcore.concepts.base import ConceptMapper, ConceptStore
    from ragcore.corpus import Corpus
    from ragcore.embedding.base import Embedder
    from ragcore.graph.base import GraphStore
    from ragcore.hierarchy.base import HierarchyStore
    from ragcore.lexical.base import LexicalStore
    from ragcore.reranking.base import Reranker
    from ragcore.stores.base import VectorStore


def default_stack(
    corpus: Corpus,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    lexical: LexicalStore | None = None,
    graph_store: GraphStore | None = None,
    concepts_store: ConceptStore | None = None,
    concept_mapper: ConceptMapper | None = None,
    hierarchy_store: HierarchyStore | None = None,
    reranker: Reranker | None = None,
    iterative: bool = False,
    judge: SufficiencyJudge | None = None,
    max_hops: int = 3,
    mmr_lambda: float = 0.7,
) -> Retriever:
    """Builds the full stack, including only the layers whose backing store
    was actually provided. `embedder`/`store` default to `corpus.embedder`/
    `corpus.store` — pass them explicitly only to use something other than
    the corpus's own configured pair (rare).

    `iterative` defaults to False: IterativeRetriever's default
    ClaudeSufficiencyJudge eagerly constructs an `anthropic.Anthropic()`
    client (needing the `generation` extra and an API key), so multi-hop is
    opt-in here the same way every other LLM-backed layer is — pass
    `iterative=True` (optionally with your own `judge`) to enable it.
    """
    embedder = embedder or corpus.embedder
    store = store or corpus.store

    retriever: Retriever
    if lexical is not None:
        retriever = HybridRetriever(embedder, store, lexical, corpus)
    else:
        retriever = VectorRetriever(embedder, store)

    if graph_store is not None:
        from ragcore.retrieval.graph import GraphRetriever

        retriever = GraphRetriever(retriever, graph_store, corpus)

    if concepts_store is not None:
        from ragcore.retrieval.concept import ConceptRetriever

        retriever = ConceptRetriever(retriever, concepts_store, corpus, embedder, concept_mapper)

    if hierarchy_store is not None:
        from ragcore.retrieval.hierarchy import HierarchyRetriever

        retriever = HierarchyRetriever(retriever, hierarchy_store, embedder)

    if iterative or judge is not None:
        retriever = IterativeRetriever(retriever, judge, max_hops=max_hops)

    if reranker is not None:
        retriever = CrossEncoderReranker(retriever, reranker)

    return MMRReranker(retriever, lambda_=mmr_lambda)
