"""HierarchyRetriever: pulls in relevant section/chapter/book summaries as
synthetic chunks alongside the inner retriever's own hits, scored with a
real cosine-derived score, and stamps the real book's source_id (unlike a
community summary's synthetic cross-source id).
"""

from __future__ import annotations

from pathlib import Path

from ragcore.corpus import Corpus
from ragcore.hierarchy.sqlite_store import SqliteHierarchyStore
from ragcore.hierarchy.types import HierarchyNode
from ragcore.retrieval.hierarchy import HierarchyRetriever
from ragcore.types import Retrieved
from tests.fakes import FakeEmbedder, InMemoryStore


class _FixedRetriever:
    def __init__(self, results: list[Retrieved]) -> None:
        self._results = results

    def retrieve(self, query, *, k=10, filters=None):
        return self._results[:k]


def _corpus(md_book: Path) -> Corpus:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    return corpus


def test_no_stored_nodes_still_returns_a_full_k(md_book: Path) -> None:
    corpus = _corpus(md_book)
    all_chunks = corpus.store.all_chunks()
    seeds = [Retrieved(chunk=c, score=1.0 - i * 0.1) for i, c in enumerate(all_chunks)]
    store = SqliteHierarchyStore(":memory:")

    retriever = HierarchyRetriever(_FixedRetriever(seeds), store, FakeEmbedder(dim=8))
    results = retriever.retrieve("anything", k=len(seeds))

    assert len(results) == len(seeds)


def test_relevant_node_arrives_as_a_synthetic_chunk_with_the_real_source_id(md_book: Path) -> None:
    corpus = _corpus(md_book)
    all_chunks = corpus.store.all_chunks()
    seed = Retrieved(chunk=all_chunks[0], score=0.1)
    embedder = FakeEmbedder(dim=8)
    query = "a query about the book"

    store = SqliteHierarchyStore(":memory:")
    real_source_id = all_chunks[0].meta.source_id
    node = HierarchyNode(
        node_id=f"hierarchy:chapter:{real_source_id}:0", level="chapter", source_id=real_source_id,
        label="Chapter One", summary="A synthesized chapter summary.",
        child_ids=(all_chunks[0].chunk_id,), source_chunk_ids=(all_chunks[0].chunk_id,),
        model="m", prompt_version="v1", created_at="now",
        vector=tuple(embedder.embed_query(query)),
    )
    store.add_node(node, fingerprint="fp")

    retriever = HierarchyRetriever(_FixedRetriever([seed]), store, embedder, max_nodes=1)
    results = retriever.retrieve(query, k=5)

    node_result = next(
        (r for r in results if r.chunk.meta.extra.get("ragcore_kind") == "hierarchy_summary"), None
    )
    assert node_result is not None
    assert node_result.chunk.chunk_id == node.node_id
    assert node_result.chunk.meta.source_id == real_source_id  # real source_id, not a synthetic one
    assert node_result.chunk.meta.extra["hierarchy_level"] == "chapter"
    assert 0.0 < node_result.score <= 1.0


def test_tenant_id_filter_excludes_another_tenants_node(md_book: Path) -> None:
    """Before tenant_id existed on HierarchyNode, nothing downstream of
    HierarchyRetriever re-checked a filter against its synthetic node
    results (unlike HybridRetriever's lexical-only hits, which do get a
    Python post-filter) - this was a real cross-tenant leak, not just a
    recall-at-scale concern. Confirms it's closed."""
    corpus = _corpus(md_book)
    all_chunks = corpus.store.all_chunks()
    seed = Retrieved(chunk=all_chunks[0], score=0.1)
    embedder = FakeEmbedder(dim=8)
    query = "a query about the book"
    real_source_id = all_chunks[0].meta.source_id

    store = SqliteHierarchyStore(":memory:")
    store.add_node(
        HierarchyNode(
            node_id=f"hierarchy:chapter:{real_source_id}:0", level="chapter", source_id=real_source_id,
            label="Chapter One", summary="tenant-b's chapter summary.",
            child_ids=(all_chunks[0].chunk_id,), source_chunk_ids=(all_chunks[0].chunk_id,),
            model="m", prompt_version="v1", created_at="now",
            vector=tuple(embedder.embed_query(query)), tenant_id="tenant-b",
        ),
        fingerprint="fp",
    )

    retriever = HierarchyRetriever(_FixedRetriever([seed]), store, embedder, max_nodes=1)
    results = retriever.retrieve(query, k=5, filters={"tenant_id": "tenant-a"})

    assert not any(r.chunk.meta.extra.get("ragcore_kind") == "hierarchy_summary" for r in results)


def test_tenant_id_filter_includes_this_tenants_own_node(md_book: Path) -> None:
    corpus = _corpus(md_book)
    all_chunks = corpus.store.all_chunks()
    seed = Retrieved(chunk=all_chunks[0], score=0.1)
    embedder = FakeEmbedder(dim=8)
    query = "a query about the book"
    real_source_id = all_chunks[0].meta.source_id

    store = SqliteHierarchyStore(":memory:")
    node = HierarchyNode(
        node_id=f"hierarchy:chapter:{real_source_id}:0", level="chapter", source_id=real_source_id,
        label="Chapter One", summary="tenant-a's chapter summary.",
        child_ids=(all_chunks[0].chunk_id,), source_chunk_ids=(all_chunks[0].chunk_id,),
        model="m", prompt_version="v1", created_at="now",
        vector=tuple(embedder.embed_query(query)), tenant_id="tenant-a",
    )
    store.add_node(node, fingerprint="fp")

    retriever = HierarchyRetriever(_FixedRetriever([seed]), store, embedder, max_nodes=1)
    results = retriever.retrieve(query, k=5, filters={"tenant_id": "tenant-a"})

    node_result = next(
        (r for r in results if r.chunk.meta.extra.get("ragcore_kind") == "hierarchy_summary"), None
    )
    assert node_result is not None
    assert node_result.chunk.meta.tenant_id == "tenant-a"


def test_respects_k(md_book: Path) -> None:
    corpus = _corpus(md_book)
    all_chunks = corpus.store.all_chunks()
    seeds = [Retrieved(chunk=c, score=1.0 - i * 0.1) for i, c in enumerate(all_chunks)]
    store = SqliteHierarchyStore(":memory:")

    retriever = HierarchyRetriever(_FixedRetriever(seeds), store, FakeEmbedder(dim=8))

    assert retriever.retrieve("q", k=0) == []
    assert len(retriever.retrieve("q", k=1)) == 1
