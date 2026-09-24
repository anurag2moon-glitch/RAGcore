"""default_stack: only wraps the layers whose backing store was actually
provided, and never eagerly constructs an LLM client (IterativeRetriever's
ClaudeSufficiencyJudge) unless multi-hop is explicitly opted into.
"""

from __future__ import annotations

from pathlib import Path

from ragcore.corpus import Corpus
from ragcore.lexical.fts5 import Fts5LexicalStore
from ragcore.retrieval.graph import GraphRetriever
from ragcore.retrieval.hybrid import HybridRetriever
from ragcore.retrieval.iterative import IterativeRetriever
from ragcore.retrieval.mmr import MMRReranker
from ragcore.retrieval.stack import default_stack
from ragcore.retrieval.vector import VectorRetriever
from tests.fakes import FakeEmbedder, InMemoryStore


def _corpus(md_book: Path) -> Corpus:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    return corpus


def test_minimal_stack_is_just_mmr_over_plain_vector_retrieval(md_book: Path) -> None:
    corpus = _corpus(md_book)

    retriever = default_stack(corpus)

    assert isinstance(retriever, MMRReranker)
    assert isinstance(retriever._inner, VectorRetriever)


def test_lexical_store_upgrades_the_base_to_hybrid_retriever(md_book: Path) -> None:
    corpus = _corpus(md_book)
    lexical = Fts5LexicalStore(":memory:")

    retriever = default_stack(corpus, lexical=lexical)

    assert isinstance(retriever._inner, HybridRetriever)


def test_graph_store_adds_graph_retriever_layer(md_book: Path) -> None:
    from ragcore.graph.sqlite_store import SqliteGraphStore

    corpus = _corpus(md_book)
    graph_store = SqliteGraphStore(":memory:")

    retriever = default_stack(corpus, graph_store=graph_store)

    assert isinstance(retriever._inner, GraphRetriever)


def test_hierarchy_store_adds_hierarchy_retriever_layer(md_book: Path) -> None:
    from ragcore.hierarchy.sqlite_store import SqliteHierarchyStore
    from ragcore.retrieval.hierarchy import HierarchyRetriever

    corpus = _corpus(md_book)
    hierarchy_store = SqliteHierarchyStore(":memory:")

    retriever = default_stack(corpus, hierarchy_store=hierarchy_store)

    assert isinstance(retriever._inner, HierarchyRetriever)


def test_iterative_layer_is_opt_in_and_never_constructed_by_default(md_book: Path) -> None:
    corpus = _corpus(md_book)

    default_retriever = default_stack(corpus)
    assert not isinstance(default_retriever._inner, IterativeRetriever)

    class _FakeJudge:
        model = "fake"

        def assess(self, original_query, hops_so_far, context):
            from ragcore.retrieval.iterative import Sufficiency

            return Sufficiency(sufficient=True, next_query=None, reasoning="r")

    with_iterative = default_stack(corpus, judge=_FakeJudge())
    assert isinstance(with_iterative._inner, IterativeRetriever)


def test_the_full_stack_actually_retrieves_something(md_book: Path) -> None:
    from ragcore.graph.sqlite_store import SqliteGraphStore

    corpus = _corpus(md_book)
    lexical = Fts5LexicalStore(":memory:")
    from ragcore.lexical.pipeline import build_lexical_index

    build_lexical_index(corpus, lexical)
    graph_store = SqliteGraphStore(":memory:")

    retriever = default_stack(corpus, lexical=lexical, graph_store=graph_store)
    corpus.retriever = retriever

    results = corpus.retrieve("structural chunking", k=3)
    assert results
