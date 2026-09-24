"""Incremental ingestion with basic versioning/rollback: detect new,
changed, or removed documents against a manifest; update only the affected
stores; snapshot before a risky sync and roll back a bad one.

    from ragcore.versioning import (
        IngestionManifest, hash_file,
        plan_sync, sync_corpus,
        create_snapshot, rollback,
        SyncPlan, SyncReport, Snapshot, SourceState,
    )
"""

from ragcore.versioning.manifest import IngestionManifest, hash_file
from ragcore.versioning.snapshot import create_snapshot, rollback
from ragcore.versioning.sync import plan_sync, sync_corpus
from ragcore.versioning.types import Snapshot, SourceState, SyncPlan, SyncReport

__all__ = [
    "IngestionManifest",
    "hash_file",
    "plan_sync",
    "sync_corpus",
    "create_snapshot",
    "rollback",
    "SyncPlan",
    "SyncReport",
    "Snapshot",
    "SourceState",
]
