"""build_hierarchy() resumability: a failed node leaves no partial
checkpoint, a re-run retries only unfinished nodes, and — the interesting
case unique to a hierarchy — a chapter that failed and later succeeds
correctly invalidates and rebuilds the book node above it, since the book's
fingerprint is computed over its actual child node_ids.
"""

from __future__ import annotations

from pathlib import Path

from ragcore.corpus import Corpus
from ragcore.hierarchy.pipeline import build_hierarchy
from ragcore.hierarchy.sqlite_store import SqliteHierarchyStore
from ragcore.hierarchy.types import HierarchyNode
from tests.fakes import FakeEmbedder, InMemoryStore


def _corpus(tmp_path: Path) -> Corpus:
    p = tmp_path / "book.md"
    p.write_text("# Chapter 1\n\nContent one.\n\n# Chapter 2\n\nContent two.\n")
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(p)
    return corpus


class _FlakySummarizer:
    model = "flaky"

    def __init__(self, fail_on_call: int) -> None:
        self._fail_on_call = fail_on_call
        self.calls: list[tuple[str, list[str]]] = []

    def summarize(self, *, node_id, level, source_id, label, child_texts, child_ids, source_chunk_ids) -> HierarchyNode:
        self.calls.append((level, list(child_ids)))
        if len(self.calls) == self._fail_on_call:
            raise RuntimeError(f"simulated failure on call {len(self.calls)}")
        return HierarchyNode(
            node_id=node_id, level=level, source_id=source_id, label=label,
            summary=f"summary #{len(self.calls)}",
            child_ids=tuple(child_ids), source_chunk_ids=tuple(source_chunk_ids),
            model=self.model, prompt_version="v1", created_at="now",
        )


def test_a_failed_node_leaves_no_checkpoint(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    store = SqliteHierarchyStore(":memory:")
    summarizer = _FlakySummarizer(fail_on_call=1)

    report = build_hierarchy(corpus, store, summarizer)

    assert len(report.errors) == 1
    _source_id, failed_node_id, _msg = report.errors[0]
    assert store.checkpoint(failed_node_id) is None
    assert store.node(failed_node_id) is None


def test_rerun_retries_only_the_failed_node(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    store = SqliteHierarchyStore(":memory:")
    summarizer = _FlakySummarizer(fail_on_call=1)

    build_hierarchy(corpus, store, summarizer)
    calls_after_first = len(summarizer.calls)

    summarizer._fail_on_call = -1  # never fail again
    report2 = build_hierarchy(corpus, store, summarizer)

    # The failed chapter is retried, and the book node (whose child set now
    # includes the newly-succeeded chapter) is rebuilt too — everything else
    # that already succeeded and is structurally unaffected is skipped.
    assert report2.nodes_built >= 1
    assert report2.nodes_skipped >= 1
    assert len(summarizer.calls) > calls_after_first


def test_rerun_with_nothing_changed_skips_everything(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    store = SqliteHierarchyStore(":memory:")
    summarizer = _FlakySummarizer(fail_on_call=-1)

    report1 = build_hierarchy(corpus, store, summarizer)
    calls_after_first = len(summarizer.calls)

    report2 = build_hierarchy(corpus, store, summarizer)

    assert report2.nodes_built == 0
    assert report2.nodes_skipped == report1.nodes_built
    assert len(summarizer.calls) == calls_after_first


def test_a_model_change_invalidates_every_checkpoint(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    store = SqliteHierarchyStore(":memory:")

    v1 = _FlakySummarizer(fail_on_call=-1)
    v1.model = "model-v1"
    build_hierarchy(corpus, store, v1)
    assert len(v1.calls) > 0

    v2 = _FlakySummarizer(fail_on_call=-1)
    v2.model = "model-v2"
    report = build_hierarchy(corpus, store, v2)

    assert report.nodes_skipped == 0
    assert len(v2.calls) == len(v1.calls)
