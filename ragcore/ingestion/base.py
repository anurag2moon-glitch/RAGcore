"""IngestCheckpointStore protocol: per-ChunkGroup resumability for
Corpus.ingest() itself, plus dead-letter tracking for Corpus.ingest_many().

Opt-in — see ragcore/ingestion/__init__.py and Corpus.__init__'s
`checkpoint_store` parameter for why this isn't the default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class DeadLetter:
    """A path that failed ingest_many() repeatedly and was given up on."""

    path: str
    error: str
    attempts: int
    updated_at: str


@runtime_checkable
class IngestCheckpointStore(Protocol):
    def checkpoint(self, source_id: str, group_id: str) -> str | None:
        """The fingerprint recorded the last time this group finished
        successfully, or None if it has never completed."""
        ...

    def mark_group_done(self, *, source_id: str, group_id: str, fingerprint: str) -> None:
        """Record that `group_id`'s chunks have been embedded and upserted
        under `fingerprint`. Must only be called after the upsert succeeds —
        see ragcore/ingestion/sqlite_store.py's module docstring for why
        ordering (upsert, then checkpoint) rather than the reverse is the
        one that can never leave a checkpoint pointing at missing chunks."""
        ...

    def mark_group_failed(
        self, *, source_id: str, group_id: str, fingerprint: str, error: str
    ) -> None:
        """Record a failed attempt. Does not clear a prior 'done' row for a
        different fingerprint — a transient failure on an unrelated group
        must not un-checkpoint groups that already succeeded."""
        ...

    def remove_source(self, source_id: str) -> None:
        """Delete every checkpoint row for `source_id` — used when a source
        is fully removed (ragcore.versioning.sync.sync_corpus), so a later
        re-ingest of a different file that happens to collide is impossible
        to confuse with stale state (it can't, since source_id is
        content-addressed, but removal should still leave no orphan rows)."""
        ...

    def mark_dead_letter(self, *, path: str, error: str, attempts: int) -> None:
        """Record that `path` failed ingest_many() `attempts` times in a row
        and will not be retried automatically."""
        ...

    def dead_letters(self) -> list[DeadLetter]:
        """Every path currently dead-lettered, for an operator to inspect
        and decide whether to fix the input and clear it."""
        ...

    def clear_dead_letter(self, path: str) -> None:
        """Remove `path` from the dead-letter list — call after a fix, so
        the next ingest_many() attempts it again instead of skipping it."""
        ...
