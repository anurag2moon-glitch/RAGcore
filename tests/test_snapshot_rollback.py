"""create_snapshot / rollback: the orchestration layer tying together
LanceDB's native tag/restore for the vector store and the stdlib sqlite3
backup API (in both directions) for every other configured store, plus the
pending/complete atomicity guard — rollback must refuse a snapshot whose
backups were never completed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragcore.corpus import Corpus
from ragcore.errors import RagcoreError
from ragcore.graph.sqlite_store import SqliteGraphStore
from ragcore.graph.types import Entity
from ragcore.versioning.manifest import IngestionManifest
from ragcore.versioning.snapshot import create_snapshot, rollback
from tests.fakes import FakeEmbedder, InMemoryStore


def test_rollback_refuses_a_pending_snapshot(tmp_path: Path) -> None:
    manifest = IngestionManifest(":memory:")
    manifest.begin_snapshot("half-taken")  # never completed

    with pytest.raises(RagcoreError, match="not 'complete'"):
        rollback(manifest, "half-taken", vector_store=InMemoryStore(dim=2))


def test_rollback_raises_for_an_unknown_tag() -> None:
    manifest = IngestionManifest(":memory:")

    with pytest.raises(RagcoreError, match="no snapshot recorded"):
        rollback(manifest, "nope", vector_store=InMemoryStore(dim=2))


def test_create_snapshot_gracefully_skips_a_vector_store_with_no_versioning_support(tmp_path: Path) -> None:
    """InMemoryStore (a test double, no LanceDB-style versioning) must not
    make create_snapshot raise — versioning is duck-typed and opt-in per
    store, not a hard requirement."""
    manifest = IngestionManifest(":memory:")

    snap = create_snapshot(
        "tag1", manifest, InMemoryStore(dim=2), snapshot_dir=tmp_path / "snaps"
    )

    assert snap.status == "complete"
    assert snap.lancedb_version is None


def test_create_snapshot_backs_up_every_provided_sqlite_store(tmp_path: Path) -> None:
    manifest = IngestionManifest(":memory:")
    graph_store = SqliteGraphStore(":memory:")
    graph_store.add_entities(
        [Entity(entity_id="entity:person:a", name="A", entity_type="person", source_chunk_ids=("s:0",))]
    )

    snap = create_snapshot(
        "tag1", manifest, InMemoryStore(dim=2), snapshot_dir=tmp_path / "snaps", graph_store=graph_store
    )

    assert "graph" in snap.backups
    assert Path(snap.backups["graph"]).exists()


def test_rollback_restores_a_sqlite_store_from_its_backup(tmp_path: Path) -> None:
    manifest = IngestionManifest(":memory:")
    graph_store = SqliteGraphStore(":memory:")
    graph_store.add_entities(
        [Entity(entity_id="entity:person:a", name="A", entity_type="person", source_chunk_ids=("s:0",))]
    )

    create_snapshot(
        "before-bad", manifest, InMemoryStore(dim=2), snapshot_dir=tmp_path / "snaps", graph_store=graph_store
    )

    # simulate a "bad ingest" that adds more data
    graph_store.add_entities(
        [Entity(entity_id="entity:person:b", name="B", entity_type="person", source_chunk_ids=("s:1",))]
    )
    assert len(graph_store.entities_by_type("person")) == 2

    rollback(manifest, "before-bad", InMemoryStore(dim=2), graph_store=graph_store)

    assert [e.entity_id for e in graph_store.entities_by_type("person")] == ["entity:person:a"]


def test_full_snapshot_rollback_round_trip_with_real_lancedb(tmp_path: Path) -> None:
    lancedb = pytest.importorskip("lancedb")  # noqa: F841
    from ragcore.stores.lancedb import LanceDBStore

    embedder = FakeEmbedder(dim=8)
    store = LanceDBStore(tmp_path / "db.lancedb", dim=8)
    corpus = Corpus(store, embedder)
    manifest = IngestionManifest(":memory:")

    p = tmp_path / "book.md"
    p.write_text("# Chapter 1\n\nGood content.\n")
    corpus.ingest(p)

    snap = create_snapshot("before-bad", manifest, store, snapshot_dir=tmp_path / "snaps")
    assert snap.lancedb_version is not None
    count_before = corpus.count()

    p2 = tmp_path / "bad.md"
    p2.write_text("# Bad\n\nMistake.\n")
    corpus.ingest(p2)
    assert corpus.count() > count_before

    rollback(manifest, "before-bad", store)

    assert corpus.count() == count_before
