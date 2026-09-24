"""LanceDBStore.snapshot/restore_snapshot/list_snapshots/drop_snapshot: the
real LanceDB tag/restore API, exercised end to end — a tag survives a
restore, restore is a rollback that appends a new version rather than
destroying history, and dropping a tag doesn't touch row data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("lancedb")

from ragcore.stores.lancedb import LanceDBStore
from ragcore.types import Chunk, ChunkMeta


def _chunk(chunk_id: str, text: str, vector: list[float]) -> Chunk:
    meta = ChunkMeta(
        source_id="s", source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=None, section=None,
        page_start=None, page_end=None, chunk_index=0, context_group_id="g",
        char_start=0, char_end=len(text), token_count=1,
    )
    return Chunk(chunk_id=chunk_id, text=text, meta=meta, vector=vector)


def test_snapshot_and_restore_round_trips_row_count(tmp_path: Path) -> None:
    store = LanceDBStore(tmp_path / "db.lancedb", dim=2)
    store.upsert([_chunk("c1", "hello", [1.0, 0.0])])
    assert store.count() == 1

    tagged_version = store.snapshot("before-bad-ingest")

    store.upsert([_chunk("c2", "world", [0.0, 1.0])])
    assert store.count() == 2

    new_version = store.restore_snapshot("before-bad-ingest")

    assert store.count() == 1
    assert store.get(["c1"])
    assert store.get(["c2"]) == []
    assert new_version > tagged_version  # restore is a NEW version, history isn't rewritten


def test_list_snapshots_reports_tag_and_version(tmp_path: Path) -> None:
    store = LanceDBStore(tmp_path / "db.lancedb", dim=2)
    store.upsert([_chunk("c1", "hello", [1.0, 0.0])])
    version = store.snapshot("v1")

    snapshots = store.list_snapshots()

    assert ("v1", version) in snapshots


def test_drop_snapshot_removes_the_tag_without_touching_data(tmp_path: Path) -> None:
    store = LanceDBStore(tmp_path / "db.lancedb", dim=2)
    store.upsert([_chunk("c1", "hello", [1.0, 0.0])])
    store.snapshot("temp-tag")

    store.drop_snapshot("temp-tag")

    assert store.list_snapshots() == []
    assert store.count() == 1  # row data untouched


def test_restoring_an_older_snapshot_after_a_later_one_still_works(tmp_path: Path) -> None:
    store = LanceDBStore(tmp_path / "db.lancedb", dim=2)
    store.upsert([_chunk("c1", "one", [1.0, 0.0])])
    store.snapshot("after-c1")

    store.upsert([_chunk("c2", "two", [0.0, 1.0])])
    store.snapshot("after-c2")

    store.upsert([_chunk("c3", "three", [1.0, 1.0])])
    assert store.count() == 3

    store.restore_snapshot("after-c1")

    assert store.count() == 1
    # both tags must still be usable — restore never rewrites history
    assert {tag for tag, _ in store.list_snapshots()} == {"after-c1", "after-c2"}
