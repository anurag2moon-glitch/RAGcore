"""Opt-in mid-ingest resumability for Corpus.ingest()/ingest_many().

Every other checkpointed pass in this codebase (concepts, contextual,
hierarchy, claims) is a separate, opt-in construction step that runs after
Corpus.ingest() — see e.g. ragcore/contextual/pipeline.py's module
docstring. Corpus.ingest() itself had no equivalent: a large file failing
mid-embed re-ran load -> chunk -> embed from scratch on retry. This module
adds the same checkpoint shape (fingerprint over content, resumed by
default, one row per unit of work) to that path too, but — unlike the
add-on passes — it is wired through an optional Corpus constructor
argument (`checkpoint_store`) rather than being always-on, because making
it always-on would change Corpus.ingest()'s upsert behavior for every
caller: resumability requires upserting each ChunkGroup as soon as it's
embedded instead of collecting every chunk into one final upsert call,
which is an observable behavior change (partial writes become visible
mid-ingest) that only a caller who asked for resumability should opt into.
See docs/adr/0002-ingest-checkpointing-opt-in.md.
"""

from __future__ import annotations

from ragcore.ingestion.base import DeadLetter, IngestCheckpointStore
from ragcore.ingestion.sqlite_store import SqliteIngestCheckpointStore

__all__ = [
    "DeadLetter",
    "IngestCheckpointStore",
    "SqliteIngestCheckpointStore",
]
