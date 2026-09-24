"""IngestionManifest: source tracking (record/mark_removed/lookup) and
snapshot tracking (begin/complete, the pending/complete distinction)."""

from __future__ import annotations

from pathlib import Path

from ragcore.versioning.manifest import IngestionManifest, hash_file


def test_hash_file_is_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("hello")
    h1 = hash_file(p)
    assert hash_file(p) == h1

    p.write_text("hello world")
    assert hash_file(p) != h1


def test_record_ingest_and_source_id_for_path_round_trip() -> None:
    manifest = IngestionManifest(":memory:")

    manifest.record_ingest(path="/books/a.md", source_id="hash1", content_hash="c1", chunk_count=5)

    assert manifest.source_id_for_path("/books/a.md") == "hash1"
    state = manifest.current_state()["/books/a.md"]
    assert state.source_id == "hash1"
    assert state.content_hash == "c1"
    assert state.chunk_count == 5


def test_record_ingest_overwrites_on_the_same_path() -> None:
    manifest = IngestionManifest(":memory:")
    manifest.record_ingest(path="/books/a.md", source_id="hash1", content_hash="c1", chunk_count=5)

    manifest.record_ingest(path="/books/a.md", source_id="hash2", content_hash="c2", chunk_count=7)

    assert manifest.source_id_for_path("/books/a.md") == "hash2"
    assert len(manifest.current_state()) == 1


def test_mark_removed_excludes_a_path_from_current_state_and_lookup() -> None:
    manifest = IngestionManifest(":memory:")
    manifest.record_ingest(path="/books/a.md", source_id="hash1", content_hash="c1", chunk_count=5)

    manifest.mark_removed("/books/a.md")

    assert manifest.source_id_for_path("/books/a.md") is None
    assert "/books/a.md" not in manifest.current_state()


def test_snapshot_lifecycle_pending_then_complete() -> None:
    manifest = IngestionManifest(":memory:")

    manifest.begin_snapshot("tag1")
    assert manifest.get_snapshot("tag1").status == "pending"

    snap = manifest.complete_snapshot("tag1", lancedb_version=3, backups={"graph": "/tmp/x.sqlite3"})

    assert snap.status == "complete"
    assert snap.lancedb_version == 3
    assert snap.backups == {"graph": "/tmp/x.sqlite3"}
    assert manifest.get_snapshot("tag1").status == "complete"


def test_get_snapshot_returns_none_for_unknown_tag() -> None:
    manifest = IngestionManifest(":memory:")
    assert manifest.get_snapshot("nope") is None


def test_list_snapshots_returns_every_recorded_snapshot() -> None:
    manifest = IngestionManifest(":memory:")
    manifest.begin_snapshot("tag1")
    manifest.complete_snapshot("tag1", lancedb_version=1, backups={})
    manifest.begin_snapshot("tag2")
    manifest.complete_snapshot("tag2", lancedb_version=2, backups={})

    tags = {s.tag for s in manifest.list_snapshots()}

    assert tags == {"tag1", "tag2"}


def test_backup_to_produces_a_restorable_copy(tmp_path: Path) -> None:
    manifest = IngestionManifest(":memory:")
    manifest.record_ingest(path="/a.md", source_id="h1", content_hash="c1", chunk_count=1)

    backup_path = tmp_path / "backup.sqlite3"
    manifest.backup_to(backup_path)

    restored = IngestionManifest(backup_path)
    assert restored.source_id_for_path("/a.md") == "h1"
