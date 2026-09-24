"""build_hierarchy: section/chapter/book node construction, the degenerate
no-sub-section chapter shortcut, and content-addressed-by-checkpoint
resumability (positional node ids, fingerprint decides rebuild vs skip).
"""

from __future__ import annotations

from pathlib import Path

from ragcore.corpus import Corpus
from ragcore.hierarchy.pipeline import build_hierarchy
from ragcore.hierarchy.sqlite_store import SqliteHierarchyStore
from ragcore.hierarchy.types import HierarchyNode
from tests.fakes import FakeEmbedder, InMemoryStore


class _FakeSummarizer:
    model = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def summarize(self, *, node_id, level, source_id, label, child_texts, child_ids, source_chunk_ids) -> HierarchyNode:
        self.calls.append((level, list(child_ids)))
        return HierarchyNode(
            node_id=node_id, level=level, source_id=source_id, label=label,
            summary=f"summary of {level} {label!r} from {len(child_texts)} parts",
            child_ids=tuple(child_ids), source_chunk_ids=tuple(source_chunk_ids),
            model=self.model, prompt_version="v1", created_at="now",
        )


def _ingest_sectioned_book(tmp_path: Path) -> Corpus:
    p = tmp_path / "book.md"
    p.write_text(
        "# Chapter 1\n\n"
        "## Section A\nText about section A.\n\n"
        "## Section B\nText about section B.\n\n"
        "# Chapter 2\n\n"
        "Flat chapter with no sub-sections, just one paragraph.\n"
    )
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(p)
    return corpus


def test_a_chapter_with_distinct_sections_gets_section_nodes(tmp_path: Path) -> None:
    corpus = _ingest_sectioned_book(tmp_path)
    store = SqliteHierarchyStore(":memory:")

    build_hierarchy(corpus, store, _FakeSummarizer())

    section_nodes = store.nodes_by_level("section")
    assert len(section_nodes) == 2
    assert {n.label for n in section_nodes} == {"Section A", "Section B"}


def test_a_chapter_with_no_sub_sections_skips_straight_to_chapter_level(tmp_path: Path) -> None:
    corpus = _ingest_sectioned_book(tmp_path)
    store = SqliteHierarchyStore(":memory:")

    build_hierarchy(corpus, store, _FakeSummarizer())

    chapter_2_chunks = [c for c in corpus.store.all_chunks() if c.meta.chapter_index == 1]
    chapter_2_id = f"hierarchy:chapter:{chapter_2_chunks[0].meta.source_id}:1"
    chapter_2_node = store.node(chapter_2_id)

    assert chapter_2_node is not None
    # its children are the chapter's own chunk_ids directly, not section nodes
    assert set(chapter_2_node.child_ids) == {c.chunk_id for c in chapter_2_chunks}
    assert store.nodes_for_source(chapter_2_chunks[0].meta.source_id)
    section_nodes_for_ch2 = [
        n for n in store.nodes_by_level("section") if n.node_id.split(":")[-2] == "1"
    ]
    assert section_nodes_for_ch2 == []


def test_chapter_node_summarizes_its_section_summaries_not_raw_chunks(tmp_path: Path) -> None:
    corpus = _ingest_sectioned_book(tmp_path)
    store = SqliteHierarchyStore(":memory:")
    summarizer = _FakeSummarizer()

    build_hierarchy(corpus, store, summarizer)

    chapter_calls = [ids for level, ids in summarizer.calls if level == "chapter"]
    section_node_ids = {n.node_id for n in store.nodes_by_level("section")}
    # Chapter 1's call should have been given section node_ids as children,
    # not chunk_ids — chapter 2's call (no sections) should have chunk_ids.
    assert any(set(ids) <= section_node_ids and ids for ids in chapter_calls)


def test_book_node_summarizes_chapter_summaries(tmp_path: Path) -> None:
    corpus = _ingest_sectioned_book(tmp_path)
    store = SqliteHierarchyStore(":memory:")

    build_hierarchy(corpus, store, _FakeSummarizer())

    book_nodes = store.nodes_by_level("book")
    assert len(book_nodes) == 1
    chapter_node_ids = {n.node_id for n in store.nodes_by_level("chapter")}
    assert set(book_nodes[0].child_ids) == chapter_node_ids


def test_every_built_node_has_an_embedded_vector(tmp_path: Path) -> None:
    corpus = _ingest_sectioned_book(tmp_path)
    store = SqliteHierarchyStore(":memory:")

    build_hierarchy(corpus, store, _FakeSummarizer())

    for node in store.all_nodes():
        assert node.vector is not None


def test_remove_source_deletes_every_node_and_checkpoint(tmp_path: Path) -> None:
    corpus = _ingest_sectioned_book(tmp_path)
    store = SqliteHierarchyStore(":memory:")
    source_id = corpus.sources()[0].source_id

    build_hierarchy(corpus, store, _FakeSummarizer())
    assert store.nodes_for_source(source_id)

    store.remove_source(source_id)

    assert store.nodes_for_source(source_id) == []
    book_node_id = f"hierarchy:book:{source_id}"
    assert store.checkpoint(book_node_id) is None
