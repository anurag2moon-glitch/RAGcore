"""Exception hierarchy for ragcore.

All library-raised exceptions derive from RagcoreError so consuming
projects can catch broadly (`except RagcoreError`) or narrowly by stage.
"""

from __future__ import annotations


class RagcoreError(Exception):
    """Base class for all exceptions raised by ragcore."""


class LoaderError(RagcoreError):
    """Raised when a file cannot be loaded (unsupported format, corrupt
    file, missing optional dependency for that format)."""


class ChunkingError(RagcoreError):
    """Raised when a document cannot be chunked (e.g. a single block
    exceeds the chunker's token budget with no valid split point)."""


class EmbeddingError(RagcoreError):
    """Raised when an embedding backend fails (API error, rate limit
    exhausted after retries, dimension mismatch)."""


class StoreError(RagcoreError):
    """Raised by a VectorStore or GraphStore for persistence failures,
    schema mismatches, or invalid queries."""
