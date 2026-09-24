"""Demonstrates the full concept layer end to end: ingest -> build_concepts
-> detect_communities -> build_community_summaries -> the full concept-aware
retriever stack (MMRReranker(ConceptRetriever(GraphRetriever(VectorRetriever))))
-> ask() with an opt-in VerifyingGenerator.

No API calls anywhere and no pre-built sample corpus required: a small
synthetic multi-chapter book is written to samples/ (gitignored) and ingested
with FakeEmbedder/InMemoryStore (tests.fakes), and every LLM-backed component
(concept extraction, community summarization, generation, verification) is
stood in for by a small deterministic fake defined in this script — the same
role tests/pride_prejudice_fixture.py plays for scripts/graph_demo.py. This
does require the 'concepts' extra (networkx) for detect_communities.

Usage:
    python scripts/concepts_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ragcore.concepts.communities import build_community_summaries, detect_communities
from ragcore.concepts.pipeline import build_concepts
from ragcore.concepts.sqlite_store import SqliteConceptStore
from ragcore.concepts.types import Concept, ConceptEdge, CommunitySummary
from ragcore.corpus import Corpus
from ragcore.generation.base import Answer
from ragcore.graph.pipeline import build_graph
from ragcore.graph.sqlite_store import SqliteGraphStore
from ragcore.graph.types import Entity, Relation
from ragcore.retrieval.concept import ConceptRetriever
from ragcore.retrieval.graph import GraphRetriever
from ragcore.retrieval.mmr import MMRReranker
from ragcore.types import Chunk, Retrieved
from ragcore.verification.claude import VerifyingGenerator
from ragcore.verification.types import ClaimCheck, Verification
from tests.fakes import FakeEmbedder, InMemoryStore

SAMPLES_DIR = Path("samples")
BOOK_PATH = SAMPLES_DIR / "concepts_demo_book.md"

BOOK_TEXT = """\
# Chapter 1: The Rise

The young captain's ambition consumed every decision. He believed that
unchecked ambition, pursued without restraint, would carry him to greatness
no matter the cost to those around him.

# Chapter 2: The Fall

By the second act, the captain's own unchecked ambition had isolated him
from every ally who once trusted him. His downfall came not from an enemy
but from the same drive that once made him great.

# Chapter 3: A Different Path

A rival commander chose duty over desire at every turn, and while her career
advanced more slowly, she kept the loyalty of everyone she led — a quiet
argument that restraint outlasts unchecked ambition.
"""


class FakeConceptExtractor:
    """Deterministic ConceptExtractor: one fixed concept per chapter plus a
    co-occurrence edge back to a running "ambition" concept, so the demo
    graph has both concepts and edges without any network call."""

    model = "fake-concept-extractor"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def extract(self, chunks: Sequence[Chunk]) -> tuple[list[Concept], list[ConceptEdge]]:
        self.calls.append([c.chunk_id for c in chunks])
        chunk_ids = tuple(c.chunk_id for c in chunks)
        ambition = Concept(
            concept_id="concept:consequences-of-unchecked-ambition",
            label="consequences of unchecked ambition",
            description="Pursuing ambition without restraint eventually costs the pursuer.",
            source_chunk_ids=chunk_ids,
        )
        chapter_index = chunks[0].meta.chapter_index
        if chapter_index == 2:  # Chapter 3 — the contrasting concept
            duty = Concept(
                concept_id="concept:duty-over-desire",
                label="choosing duty over desire",
                description="Restraint and loyalty are argued to outlast unchecked ambition.",
                source_chunk_ids=chunk_ids,
            )
            edge = ConceptEdge(
                edge_id="edge:concept:consequences-of-unchecked-ambition:concept:duty-over-desire",
                source_concept_id="concept:consequences-of-unchecked-ambition",
                target_concept_id="concept:duty-over-desire",
                weight=0,
                source_chunk_ids=chunk_ids,
            )
            return [ambition, duty], [edge]
        return [ambition], []


class FakeSummarizer:
    model = "fake-summarizer"

    def summarize(self, *, community_id: str, concepts, chunks) -> CommunitySummary:
        labels = ", ".join(c.label for c in concepts)
        return CommunitySummary(
            community_id=community_id,
            label=f"Cluster: {labels}",
            summary=f"These sources collectively discuss: {labels}.",
            agreements=("All three chapters treat ambition as morally weighty.",),
            divergences=("Chapters 1-2 dramatize ambition's cost; Chapter 3 argues against it directly.",),
            concept_ids=tuple(c.concept_id for c in concepts),
            source_chunk_ids=tuple(c.chunk_id for c in chunks),
            source_ids=tuple(sorted({c.meta.source_id for c in chunks})),
            model=self.model,
            prompt_version="demo-v1",
            created_at="1970-01-01T00:00:00+00:00",
        )


class FakeGenerator:
    model = "fake-generator"

    def generate(self, query: str, context: Sequence[Retrieved]) -> Answer:
        text = " ".join(r.chunk.text for r in context)[:400]
        return Answer(text=f"Based on the sources: {text}", citations=list(context))


class FakeVerifier:
    """Deterministic Verifier: everything is "supported" if any citation is
    a real passage, "supported_by_synthesis" if only summaries back it."""

    model = "fake-verifier"

    def verify(self, query: str, answer: Answer) -> Verification:
        real = any(r.chunk.meta.extra.get("ragcore_kind") != "community_summary" for r in answer.citations)
        status = "supported" if real else "supported_by_synthesis"
        check = ClaimCheck(
            claim=answer.text,
            status=status,
            support=tuple(r.chunk.chunk_id for r in answer.citations),
        )
        return Verification(checks=(check,), model=self.model)


def main() -> None:
    SAMPLES_DIR.mkdir(exist_ok=True)
    BOOK_PATH.write_text(BOOK_TEXT)

    embedder = FakeEmbedder(dim=16)
    corpus = Corpus(InMemoryStore(dim=16), embedder)
    corpus.ingest(BOOK_PATH)

    # -- 1. concept extraction (checkpointed, resumable) --------------
    concept_extractor = FakeConceptExtractor()
    concepts_store = SqliteConceptStore(":memory:")
    concept_report = build_concepts(corpus, concepts_store, concept_extractor)
    print("=== ConceptReport ===")
    print(concept_report)

    # -- 2. community detection + summarization ------------------------
    community_report = detect_communities(concepts_store, seed=0)
    print("\n=== CommunityReport ===")
    print(community_report)
    n_summaries = build_community_summaries(corpus, concepts_store, FakeSummarizer())
    print(f"community summaries computed: {n_summaries}")

    # -- 3. entity graph, for the full retriever stack -----------------
    graph_store = SqliteGraphStore(":memory:")

    class _NoopEntityExtractor:
        model = "noop"

        def extract(self, chunks: Sequence[Chunk]) -> tuple[list[Entity], list[Relation]]:
            return [], []

    build_graph(corpus, graph_store, _NoopEntityExtractor())

    # -- 4. the full retriever stack -----------------------------------
    corpus.retriever = MMRReranker(
        ConceptRetriever(
            GraphRetriever(corpus.retriever, graph_store, corpus),
            concepts_store,
            corpus,
            embedder,
        ),
        per_source_cap=2,
    )

    query = "what happens when ambition goes unchecked"
    results = corpus.retrieve(query, k=5)
    print(f"\n=== retrieve({query!r}) ===")
    print(
        "(scores aren't meaningful here: FakeEmbedder is hash-based, and "
        "tests.fakes.InMemoryStore scores by raw cosine rather than "
        "LanceDBStore's 1/(1+distance) transform ConceptRetriever/MMR "
        "assume — this demo is about wiring, not retrieval quality; see "
        "scripts/query.py for a real, semantically meaningful example)"
    )
    for r in results:
        kind = r.chunk.meta.extra.get("ragcore_kind", "passage")
        print(f"  score={r.score:.3f}  kind={kind:18}  {r.chunk.text[:70]!r}")

    # -- 5. ask() with opt-in verification ------------------------------
    corpus.generator = VerifyingGenerator(FakeGenerator(), FakeVerifier())
    answer = corpus.ask(query, k=5)
    print("\n=== ask() ===")
    print(answer.text)
    print("verification:", [(c.status, c.claim[:40]) for c in answer.verification.checks])


if __name__ == "__main__":
    main()
