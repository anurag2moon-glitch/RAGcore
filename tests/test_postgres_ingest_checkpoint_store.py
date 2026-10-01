"""PostgresIngestCheckpointStore against a real Postgres server. Excluded
by default (see pyproject.toml `addopts`); run explicitly with
`pytest -m postgres`. Requires RAGCORE_TEST_POSTGRES_URL in the
environment (a libpq connection string, e.g.
`postgresql://user:pass@localhost/ragcore_test`) — skips cleanly when
unset, the same convention tests/test_live_voyage.py and
tests/integration's `gutenberg` marker already use for an opt-in external
dependency.

Mirrors tests/test_ingest_checkpoint_store.py's cases one-for-one (minus
backup_to/restore_from_backup, which aren't part of the
IngestCheckpointStore protocol and have no Postgres equivalent in this
store — see ragcore/ingestion/postgres_store.py's module docstring),
against the real server instead of SQLite, each test using its own
uniquely-prefixed source_id/path so repeated runs against a persistent
database don't collide with leftover rows from a previous run.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.postgres

CONNINFO = os.environ.get("RAGCORE_TEST_POSTGRES_URL")
pytest.importorskip("psycopg")


@pytest.fixture
def store():
    if not CONNINFO:
        pytest.skip("RAGCORE_TEST_POSTGRES_URL not set")
    from ragcore.ingestion.postgres_store import PostgresIngestCheckpointStore

    s = PostgresIngestCheckpointStore(CONNINFO)
    yield s


@pytest.fixture
def src() -> str:
    """A unique source_id per test, so reruns against a persistent
    database never collide with a previous run's rows."""
    return f"test-src-{uuid.uuid4()}"


def test_checkpoint_round_trip(store, src: str) -> None:
    assert store.checkpoint(src, "g0") is None

    store.mark_group_done(source_id=src, group_id="g0", fingerprint="fp1")
    assert store.checkpoint(src, "g0") == "fp1"
    assert store.checkpoint(src, "g1") is None


def test_mark_group_done_is_idempotent_and_overwrites_fingerprint(store, src: str) -> None:
    store.mark_group_done(source_id=src, group_id="g0", fingerprint="fp1")
    store.mark_group_done(source_id=src, group_id="g0", fingerprint="fp2")
    assert store.checkpoint(src, "g0") == "fp2"


def test_mark_group_failed_does_not_clear_a_done_sibling(store, src: str) -> None:
    store.mark_group_done(source_id=src, group_id="g0", fingerprint="fp0")
    store.mark_group_failed(source_id=src, group_id="g1", fingerprint="fp1", error="boom")

    assert store.checkpoint(src, "g0") == "fp0"
    assert store.checkpoint(src, "g1") is None


def test_remove_source_clears_every_group(store, src: str) -> None:
    other_src = f"{src}-other"
    store.mark_group_done(source_id=src, group_id="g0", fingerprint="fp0")
    store.mark_group_done(source_id=src, group_id="g1", fingerprint="fp1")
    store.mark_group_done(source_id=other_src, group_id="g0", fingerprint="fp2")

    store.remove_source(src)

    assert store.checkpoint(src, "g0") is None
    assert store.checkpoint(src, "g1") is None
    assert store.checkpoint(other_src, "g0") == "fp2"  # untouched


def test_dead_letter_lifecycle(store, src: str) -> None:
    path = f"{src}.pdf"
    store.mark_dead_letter(path=path, error="boom", attempts=3)
    [dl] = [d for d in store.dead_letters() if d.path == path]
    assert dl.error == "boom"
    assert dl.attempts == 3

    store.mark_dead_letter(path=path, error="boom again", attempts=4)
    [dl] = [d for d in store.dead_letters() if d.path == path]
    assert dl.attempts == 4  # overwritten, not duplicated

    store.clear_dead_letter(path)
    assert not [d for d in store.dead_letters() if d.path == path]

    store.clear_dead_letter(f"{path}-never-existed")  # no-op, must not raise
