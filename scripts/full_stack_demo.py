"""Demonstrates the complete seven-layer default retriever stack end to end:

    MMRReranker(
      CrossEncoderReranker(
        IterativeRetriever(
          HierarchyRetriever(
            ConceptRetriever(
              GraphRetriever(
                HybridRetriever(embedder, store, lexical, corpus))))))))

No API calls anywhere: every LLM-backed component (entity/concept
extraction, community/hierarchy summarization, sufficiency judging,
cross-encoder reranking) is stood in for by a small deterministic fake, the
same role tests/pride_prejudice_fixture.py plays for scripts/graph_demo.py.
Requires the 'concepts' extra (networkx) for detect_communities.

Usage:
    python scripts/full_stack_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ragcore.concepts.communities import build_community_summaries, detect_communities
from ragcore.concepts.pipeline import build_concepts
from ragcore.concepts.sqlite_store import SqliteConceptStore
from ragcore.concepts.types import Concept, CommunitySummary
from ragcore.corpus import Corpus
from ragcore.graph.pipeline import build_graph
from ragcore.graph.sqlite_store import SqliteGraphStore
from ragcore.graph.types import Entity, Relation
from ragcore.hierarchy.pipeline import build_hierarchy
from ragcore.hierarchy.sqlite_store import SqliteHierarchyStore
from ragcore.hierarchy.types import HierarchyNode
from ragcore.lexical.fts5 import Fts5LexicalStore
from ragcore.lexical.pipeline import build_lexical_index
from ragcore.reranking.base import RerankHit
from ragcore.retrieval.iterative import Sufficiency
from ragcore.retrieval.stack import default_stack
from ragcore.types import Chunk
from tests.fakes import FakeEmbedder, InMemoryStore

SAMPLES_DIR = Path("samples")
BOOK_PATH = SAMPLES_DIR / "full_stack_demo_book.md"

BOOK_TEXT = """\
# Chapter 1: The Rise

## Early ambition

The young captain's ambition consumed every decision. He believed that
unchecked ambition, pursued without restraint, would carry him to greatness.

## First victories

His early campaigns won him allies and admiration in equal measure.

# Chapter 2: The Fall

By the second act, the captain's own unchecked ambition had isolated him
from every ally who once trusted him. His downfall came from within.
"""


class _NoopEntityExtractor:
    model = "noop"

    def extract(self, chunks: Sequence[Chunk]) -> tuple[list[Entity], list[Relation]]:
        return [], []


class _FakeConceptExtractor:
    model = "fake-concepts"

    def extract(self, chunks: Sequence[Chunk]):
        concept = Concept(
            concept_id="concept:unchecked-ambition",
            label="consequences of unchecked ambition",
            description="Ambition pursued without restraint costs the pursuer.",
            source_chunk_ids=tuple(c.chunk_id for c in chunks),
        )
        return [concept], []


class _FakeSummarizer:
    model = "fake-summarizer"

    def summarize(self, *, community_id, concepts, chunks) -> CommunitySummary:
        return CommunitySummary(
            community_id=community_id, label="Ambition", summary="A synthesized community summary.",
            agreements=(), divergences=(), concept_ids=tuple(c.concept_id for c in concepts),
            source_chunk_ids=tuple(c.chunk_id for c in chunks),
            source_ids=tuple(sorted({c.meta.source_id for c in chunks})),
            model=self.model, prompt_version="v1", created_at="now",
        )


class _FakeHierarchySummarizer:
    model = "fake-hierarchy"

    def summarize(self, *, node_id, level, source_id, label, child_texts, child_ids, source_chunk_ids) -> HierarchyNode:
        return HierarchyNode(
            node_id=node_id, level=level, source_id=source_id, label=label,
            summary=f"A synthesized {level} summary for {label!r}.",
            child_ids=tuple(child_ids), source_chunk_ids=tuple(source_chunk_ids),
            model=self.model, prompt_version="v1", created_at="now",
        )


class _FakeJudge:
    model = "fake-judge"

    def assess(self, original_query, hops_so_far, context) -> Sufficiency:
        return Sufficiency(sufficient=True, next_query=None, reasoning="fake: always sufficient on hop 0")


class _FakeReranker:
    model = "fake-reranker"

    def rerank(self, query, documents, *, top_n=None):
        # Deterministic "relevance": prefer documents mentioning "ambition".
        hits = [
            RerankHit(index=i, relevance_score=1.0 if "ambition" in doc.lower() else 0.3)
            for i, doc in enumerate(documents)
        ]
        return sorted(hits, key=lambda h: h.relevance_score, reverse=True)


def main() -> None:
    SAMPLES_DIR.mkdir(exist_ok=True)
    BOOK_PATH.write_text(BOOK_TEXT)

    embedder = FakeEmbedder(dim=16)
    corpus = Corpus(InMemoryStore(dim=16), embedder)
    corpus.ingest(BOOK_PATH)

    lexical = Fts5LexicalStore(":memory:")
    build_lexical_index(corpus, lexical)

    graph_store = SqliteGraphStore(":memory:")
    build_graph(corpus, graph_store, _NoopEntityExtractor())

    concepts_store = SqliteConceptStore(":memory:")
    build_concepts(corpus, concepts_store, _FakeConceptExtractor())
    detect_communities(concepts_store, seed=0)
    build_community_summaries(corpus, concepts_store, _FakeSummarizer())

    hierarchy_store = SqliteHierarchyStore(":memory:")
    build_hierarchy(corpus, hierarchy_store, _FakeHierarchySummarizer())

    print("=== Building the full stack ===")
    retriever = default_stack(
        corpus,
        lexical=lexical,
        graph_store=graph_store,
        concepts_store=concepts_store,
        hierarchy_store=hierarchy_store,
        iterative=True,
        judge=_FakeJudge(),
        reranker=_FakeReranker(),
    )
    # default_stack() doesn't include CrossEncoderReranker unless a
    # `reranker` is passed — it is here, so confirm the composition:
    print(type(retriever).__name__)
    print(f"  -> {type(retriever._inner).__name__}")
    print(f"    -> {type(retriever._inner._inner).__name__}")

    corpus.retriever = retriever
    query = "what happens when ambition goes unchecked"
    results = corpus.retrieve(query, k=5)

    print(f"\n=== retrieve({query!r}) ===")
    print(
        "(scores/ranking aren't semantically meaningful here — FakeEmbedder "
        "is hash-based; this demo is about verifying the seven-layer "
        "composition wires together and returns results, not retrieval "
        "quality — see scripts/query.py for a real, semantic example)"
    )
    for r in results:
        kind = r.chunk.meta.extra.get("ragcore_kind", "passage")
        hop = r.chunk.meta.extra.get("hop_index")
        ce_score = r.chunk.meta.extra.get("cross_encoder_score")
        print(f"  score={r.score:.3f}  kind={kind:18}  hop={hop}  ce_score={ce_score}  {r.chunk.text[:60]!r}")


if __name__ == "__main__":
    main()
