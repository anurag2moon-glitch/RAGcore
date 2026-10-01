"""Exception hierarchy for ragcore.

All library-raised exceptions derive from RagcoreError so consuming
projects can catch broadly (`except RagcoreError`) or narrowly by stage.
"""

from __future__ import annotations

import functools
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


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


class GenerationError(RagcoreError):
    """Raised when a Generator backend fails (API error, refusal, rate
    limit exhausted after retries)."""


def wrap_backend_errors(fn: F) -> F:
    """Decorator for a storage-layer class's public methods: catches any
    exception the method raises and re-raises it as
    `StoreError(f"{fn.__qualname__} failed: {e}") from e`, except an
    exception that's already a RagcoreError, which passes through
    unchanged (so an intentional typed raise already inside the method —
    e.g. a filter-validation StoreError — is never double-wrapped).

    Applied broadly (`except Exception`) rather than to an enumerated list
    of each backend's specific exception types (sqlite3.Error and
    subclasses; whatever lancedb/pyarrow raises) — see
    docs/adr/0004-wrap-backend-exceptions-broadly.md for why: confidently
    enumerating every exception type two different native-extension
    libraries can raise isn't something this decorator can verify, and
    missing one would silently reintroduce the exact gap it exists to
    close. `from e` always preserves the original exception via
    `__cause__`, so it's never actually lost, just re-typed.
    """

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> object:
        try:
            return fn(*args, **kwargs)
        except RagcoreError:
            raise
        except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
            raise StoreError(f"{fn.__qualname__} failed: {e}") from e

    return wrapper  # type: ignore[return-value]
