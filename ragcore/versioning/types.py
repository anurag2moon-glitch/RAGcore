"""Data types for incremental ingestion and basic versioning/rollback."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceState:
    """One file path's last-known ingested state, as tracked by
    IngestionManifest — the state Corpus itself deliberately never keeps."""

    path: str
    source_id: str
    content_hash: str
    chunk_count: int
    ingested_at: str


@dataclass(frozen=True)
class SyncPlan:
    """The result of comparing a given list of paths against the manifest's
    last-known state — computed by re-hashing each path with the same
    content-addressing scheme ragcore.types.Document.source_id already
    uses, so "has this file changed?" needs no new hashing convention."""

    added: tuple[str, ...]
    changed: tuple[str, ...]
    removed: tuple[str, ...]
    """Paths (as stored in the manifest) present in the manifest but absent
    from the given path list."""

    unchanged: tuple[str, ...]


@dataclass(frozen=True)
class SyncReport:
    added: tuple[str, ...]
    changed: tuple[str, ...]
    removed: tuple[str, ...]
    unchanged: tuple[str, ...]
    chunks_ingested: int
    errors: tuple[tuple[str, str], ...]
    """(path, error message) for every added/changed path whose ingest()
    call raised — collected, not propagated, matching every other
    best-effort batch operation in this codebase (Corpus.ingest_many)."""

    snapshot_tag: str | None
    """The tag of the pre-sync snapshot taken, if `snapshot=True` and there
    was anything to snapshot before — None otherwise (nothing changed, or
    snapshotting was disabled)."""


@dataclass(frozen=True)
class Snapshot:
    tag: str
    status: str
    """"pending" | "complete" — a snapshot is only ever rolled back to when
    "complete"; rollback refuses a "pending" one rather than silently using
    a half-taken snapshot."""

    lancedb_version: int | None
    backups: dict
    """store-name -> backup file path, for every SQLite-backed store that
    was included in this snapshot."""

    created_at: str
