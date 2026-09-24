"""create_snapshot / rollback — basic ingestion versioning: a LanceDB tag
for the vector store (native, append-only) plus stdlib-sqlite3 online
backups for every other configured store, orchestrated as one logical unit.

The LanceDB tag and the SQLite backups are not one atomic transaction —
they're two different storage engines. IngestionManifest.begin_snapshot()
records a 'pending' row before either happens; complete_snapshot() flips it
to 'complete' only after every backup has succeeded. rollback() refuses to
restore a 'pending' snapshot, so a crash mid-snapshot is detectable rather
than silently unsafe to roll back to.

Every store argument here is optional and duck-typed (`hasattr` checked,
not part of any Protocol) — versioning is an opt-in capability layered on
top of whichever stores a consumer actually configured, the same way every
other add-on in this codebase only activates for the pieces that are
present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragcore.errors import RagcoreError
from ragcore.versioning.manifest import IngestionManifest
from ragcore.versioning.types import Snapshot

DEFAULT_MAX_SNAPSHOTS = 10

_STORE_NAMES = ("graph", "concepts", "hierarchy", "context", "lexical")


def create_snapshot(
    tag: str,
    manifest: IngestionManifest,
    vector_store: Any,
    *,
    snapshot_dir: str | Path,
    graph_store: Any = None,
    concept_store: Any = None,
    hierarchy_store: Any = None,
    context_store: Any = None,
    lexical: Any = None,
    max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
) -> Snapshot:
    manifest.begin_snapshot(tag)

    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    lancedb_version = None
    if hasattr(vector_store, "snapshot"):
        lancedb_version = vector_store.snapshot(tag)
        if hasattr(vector_store, "list_snapshots") and hasattr(vector_store, "drop_snapshot"):
            _apply_retention(vector_store, max_snapshots)

    stores = dict(
        zip(_STORE_NAMES, (graph_store, concept_store, hierarchy_store, context_store, lexical), strict=True)
    )
    backups: dict[str, str] = {}
    for name, store in stores.items():
        if store is not None and hasattr(store, "backup_to"):
            backup_path = snapshot_dir / f"{tag}.{name}.sqlite3"
            store.backup_to(backup_path)
            backups[name] = str(backup_path)

    return manifest.complete_snapshot(tag, lancedb_version=lancedb_version, backups=backups)


def _apply_retention(vector_store: Any, max_snapshots: int) -> None:
    snapshots = sorted(vector_store.list_snapshots(), key=lambda pair: pair[1])  # by version, oldest first
    while len(snapshots) > max_snapshots:
        oldest_tag, _oldest_version = snapshots.pop(0)
        # Dropping the tag re-exposes that version to normal
        # cleanup_old_versions()/optimize() — the intended GC path, not a
        # direct delete here.
        vector_store.drop_snapshot(oldest_tag)


def rollback(
    manifest: IngestionManifest,
    tag: str,
    vector_store: Any,
    *,
    graph_store: Any = None,
    concept_store: Any = None,
    hierarchy_store: Any = None,
    context_store: Any = None,
    lexical: Any = None,
) -> None:
    snapshot = manifest.get_snapshot(tag)
    if snapshot is None:
        raise RagcoreError(f"no snapshot recorded with tag {tag!r}")
    if snapshot.status != "complete":
        raise RagcoreError(
            f"snapshot {tag!r} has status {snapshot.status!r}, not 'complete' — "
            "refusing to roll back to a partially-taken snapshot"
        )

    if snapshot.lancedb_version is not None and hasattr(vector_store, "restore_snapshot"):
        vector_store.restore_snapshot(tag)

    stores = dict(
        zip(_STORE_NAMES, (graph_store, concept_store, hierarchy_store, context_store, lexical), strict=True)
    )
    for name, store in stores.items():
        backup_path = snapshot.backups.get(name)
        if backup_path and store is not None and hasattr(store, "restore_from_backup"):
            store.restore_from_backup(backup_path)
