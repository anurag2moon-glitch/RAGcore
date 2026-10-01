"""Unit tests for SqliteIngestCheckpointStore — mirrors
tests/test_contextual_resume.py's style of testing the checkpoint store in
isolation from Corpus.
"""

from __future__ import annotations

from pathlib import Path

from ragcore.ingestion.sqlite_store import SqliteIngestCheckpointStore


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    store = SqliteIngestCheckpointStore(tmp_path / "checkpoints.sqlite3")
    assert store.checkpoint("src1", "g0") is None

    store.mark_group_done(source_id="src1", group_id="g0", fingerprint="fp1")
    assert store.checkpoint("src1", "g0") == "fp1"
    assert store.checkpoint("src1", "g1") is None
    assert store.checkpoint("src2", "g0") is None


def test_mark_group_done_is_idempotent_and_overwrites_fingerprint(tmp_path: Path) -> None:
    store = SqliteIngestCheckpointStore(tmp_path / "checkpoints.sqlite3")
    store.mark_group_done(source_id="src1", group_id="g0", fingerprint="fp1")
    store.mark_group_done(source_id="src1", group_id="g0", fingerprint="fp2")
    assert store.checkpoint("src1", "g0") == "fp2"


def test_mark_group_failed_does_not_clear_a_done_sibling(tmp_path: Path) -> None:
    store = SqliteIngestCheckpointStore(tmp_path / "checkpoints.sqlite3")
    store.mark_group_done(source_id="src1", group_id="g0", fingerprint="fp0")
    store.mark_group_failed(source_id="src1", group_id="g1", fingerprint="fp1", error="boom")

    assert store.checkpoint("src1", "g0") == "fp0"
    assert store.checkpoint("src1", "g1") is None  # failed, not done


def test_remove_source_clears_every_group(tmp_path: Path) -> None:
    store = SqliteIngestCheckpointStore(tmp_path / "checkpoints.sqlite3")
    store.mark_group_done(source_id="src1", group_id="g0", fingerprint="fp0")
    store.mark_group_done(source_id="src1", group_id="g1", fingerprint="fp1")
    store.mark_group_done(source_id="src2", group_id="g0", fingerprint="fp2")

    store.remove_source("src1")

    assert store.checkpoint("src1", "g0") is None
    assert store.checkpoint("src1", "g1") is None
    assert store.checkpoint("src2", "g0") == "fp2"  # untouched


def test_dead_letter_lifecycle(tmp_path: Path) -> None:
    store = SqliteIngestCheckpointStore(tmp_path / "checkpoints.sqlite3")
    assert store.dead_letters() == []

    store.mark_dead_letter(path="book.pdf", error="boom", attempts=3)
    [dl] = store.dead_letters()
    assert dl.path == "book.pdf"
    assert dl.error == "boom"
    assert dl.attempts == 3

    store.mark_dead_letter(path="book.pdf", error="boom again", attempts=4)
    [dl] = store.dead_letters()
    assert dl.attempts == 4  # overwritten, not duplicated

    store.clear_dead_letter("book.pdf")
    assert store.dead_letters() == []

    store.clear_dead_letter("never-existed.pdf")  # no-op, must not raise


def test_backup_and_restore_round_trip(tmp_path: Path) -> None:
    store = SqliteIngestCheckpointStore(tmp_path / "checkpoints.sqlite3")
    store.mark_group_done(source_id="src1", group_id="g0", fingerprint="fp0")

    backup_path = tmp_path / "backup.sqlite3"
    store.backup_to(backup_path)

    store.mark_group_done(source_id="src1", group_id="g1", fingerprint="fp1")
    assert store.checkpoint("src1", "g1") == "fp1"

    store.restore_from_backup(backup_path)
    assert store.checkpoint("src1", "g0") == "fp0"
    assert store.checkpoint("src1", "g1") is None  # not in the backup
