"""LanceDBStore.create_vector_index/compact/index_stats — the Phase 2
"index lifecycle management" gap list item. Exercised against a real,
on-disk LanceDB table (no mocking the backend, matching
test_lancedb_snapshot.py's convention) with enough rows for a real IVF
index to train.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

pytest.importorskip("lancedb")

from ragcore.errors import StoreError
from ragcore.stores.lancedb import LanceDBStore
from ragcore.types import Chunk, ChunkMeta

DIM = 8
ROW_COUNT = 300


def _chunk(i: int, vector: list[float]) -> Chunk:
    meta = ChunkMeta(
        source_id="s", source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=None, section=None,
        page_start=None, page_end=None, chunk_index=i, context_group_id="g",
        char_start=0, char_end=1, token_count=1,
    )
    return Chunk(chunk_id=f"c{i}", text=f"t{i}", meta=meta, vector=vector)


def _populated_store(tmp_path: Path, seed: int = 0) -> tuple[LanceDBStore, list[Chunk]]:
    rng = random.Random(seed)
    chunks = [_chunk(i, [rng.random() for _ in range(DIM)]) for i in range(ROW_COUNT)]
    store = LanceDBStore(tmp_path / "db.lancedb", dim=DIM)
    store.upsert(chunks)
    return store, chunks


def test_no_index_by_default(tmp_path: Path) -> None:
    store, _ = _populated_store(tmp_path)
    assert store.index_stats() == []  # brute-force search until explicitly requested


def test_create_vector_index_ivf_pq(tmp_path: Path) -> None:
    store, _ = _populated_store(tmp_path)
    store.create_vector_index(method="ivf_pq", num_partitions=4)

    [stats] = store.index_stats()
    assert stats["columns"] == ["vector"]
    assert stats["index_type"] == "IvfPq"
    assert stats["num_indexed_rows"] == ROW_COUNT
    assert stats["num_unindexed_rows"] == 0


def test_create_vector_index_hnsw_sq(tmp_path: Path) -> None:
    store, _ = _populated_store(tmp_path)
    store.create_vector_index(method="hnsw_sq", num_partitions=4)

    [stats] = store.index_stats()
    assert stats["index_type"] == "IvfHnswSq"
    assert stats["num_indexed_rows"] == ROW_COUNT


def test_create_vector_index_replace_swaps_the_index_type(tmp_path: Path) -> None:
    store, _ = _populated_store(tmp_path)
    store.create_vector_index(method="hnsw_sq", num_partitions=4)
    store.create_vector_index(method="ivf_pq", num_partitions=4, replace=True)

    [stats] = store.index_stats()
    assert stats["index_type"] == "IvfPq"


def test_query_still_finds_the_correct_nearest_neighbor_after_indexing(tmp_path: Path) -> None:
    store, chunks = _populated_store(tmp_path)
    target = chunks[42]

    before = store.query(target.vector, k=1)
    assert before[0].chunk.chunk_id == target.chunk_id

    store.create_vector_index(method="ivf_pq", num_partitions=4)

    # IVF-PQ is an *approximate* index: a query vector that happens to sit
    # near a partition boundary can occasionally miss its own exact
    # self-match at k=1 purely from quantization, even though it's still
    # the true nearest neighbor — asserting "in the top few" rather than
    # "at rank 1" is the correct robustness bar for an approximate index,
    # not a weakened test.
    after = store.query(target.vector, k=5)
    assert target.chunk_id in {r.chunk.chunk_id for r in after}


def test_compact_runs_without_error_and_preserves_row_count(tmp_path: Path) -> None:
    store, _ = _populated_store(tmp_path)
    store.create_vector_index(method="ivf_pq", num_partitions=4)

    store.compact(cleanup_older_than_days=0)

    assert store.count() == ROW_COUNT
    [stats] = store.index_stats()
    assert stats["num_indexed_rows"] == ROW_COUNT


def test_compact_refuses_to_remove_a_tagged_snapshot(tmp_path: Path) -> None:
    """Verified empirically, not assumed from LanceDB's docs: an aggressive
    cleanup_older_than_days=0 while a tag points at an old version raises
    rather than silently deleting or silently skipping it — see
    snapshot()'s and compact()'s docstrings, corrected to match this."""
    store, _ = _populated_store(tmp_path)
    store.snapshot("v1")
    store.upsert([_chunk(ROW_COUNT, [0.5] * DIM)])  # a later version

    with pytest.raises(StoreError, match="tagged"):
        store.compact(cleanup_older_than_days=0)

    # the tag must still resolve and restore correctly — the refused
    # compact() must not have left anything half-done
    store.restore_snapshot("v1")
    assert store.count() == ROW_COUNT


def test_compact_succeeds_once_the_tag_is_dropped(tmp_path: Path) -> None:
    store, _ = _populated_store(tmp_path)
    store.snapshot("v1")
    store.upsert([_chunk(ROW_COUNT, [0.5] * DIM)])
    store.drop_snapshot("v1")

    store.compact(cleanup_older_than_days=0)  # no longer protected, must not raise

    assert store.count() == ROW_COUNT + 1


def test_unindexed_rows_added_after_index_creation_are_tracked(tmp_path: Path) -> None:
    store, _ = _populated_store(tmp_path)
    store.create_vector_index(method="ivf_pq", num_partitions=4)

    store.upsert([_chunk(ROW_COUNT, [0.1] * DIM)])

    [stats] = store.index_stats()
    assert stats["num_unindexed_rows"] == 1

    store.compact()
    [stats_after] = store.index_stats()
    assert stats_after["num_unindexed_rows"] == 0
    assert stats_after["num_indexed_rows"] == ROW_COUNT + 1
