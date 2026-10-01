"""RawFileStore protocol: durable storage for the original, unmodified
file bytes behind a Document (docs/ARCHITECTURE.md Phase 2 gap list item
12: "raw file storage on S3-compatible object storage").

Deliberately separate from VectorStore (ragcore/stores/base.py) — a
Document's extracted/chunked/embedded representation and its original
file bytes have different lifecycles and different reasons to exist.
Keeping the original bytes durable matters independently of anything
VectorStore does: `Document.source_id`/every `Chunk.chunk_id` are derived
from a local file at ingest time, and re-deriving them (e.g. to re-run a
Loader against a new format version, or to recover after local disk loss)
requires the original bytes to still exist somewhere, not just their
extracted/embedded downstream representation.

Not wired into Corpus.ingest() itself — ingest() already takes a local
Path and has no opinion on where that file came from or should also be
archived to. A caller uploads before/after calling ingest() (or as part
of its own pre-ingest pipeline) and is responsible for the local Path it
hands to ingest() existing at ingest time, same as today.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class RawFileStore(Protocol):
    def put(self, source_id: str, path: Path) -> str:
        """Uploads the file at `path`, keyed by `source_id`. Returns a
        storage-backend-specific reference (e.g. an `s3://bucket/key` URI)
        a caller can persist alongside `source_id` for later retrieval —
        not required to call `get`/`exists`/`delete` again, which only
        need `source_id`."""
        ...

    def get(self, source_id: str, dest: Path) -> None:
        """Downloads the file stored under `source_id` to `dest`. Raises
        if nothing is stored under `source_id`."""
        ...

    def exists(self, source_id: str) -> bool: ...

    def delete(self, source_id: str) -> bool:
        """Returns True if something was deleted, False if nothing was
        stored under `source_id` (not an error — same idempotent-delete
        convention as VectorStore.delete)."""
        ...

    def original_filename(self, source_id: str) -> str | None:
        """The original file's name as given to `put()`, if known — lets
        a caller pick a sensible extension/name for `get()`'s `dest`
        without having tracked it separately."""
        ...
