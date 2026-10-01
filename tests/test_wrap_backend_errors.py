"""Unit tests for ragcore.errors.wrap_backend_errors in isolation — the
decorator applied to every public method of every storage-layer class
(docs/adr/0004-wrap-backend-exceptions-broadly.md). A real end-to-end
case (SqliteGraphStore + a lost WAL sidecar) is covered separately by
tests/integration/test_failure_injection.py::test_sqlite_graph_wal_sidecar_loss_raises_a_typed_store_error.
"""

from __future__ import annotations

import pytest

from ragcore.errors import LoaderError, RagcoreError, StoreError, wrap_backend_errors


def test_wraps_an_arbitrary_exception_as_store_error() -> None:
    @wrap_backend_errors
    def boom() -> None:
        raise RuntimeError("disk I/O error")

    with pytest.raises(StoreError, match="disk I/O error"):
        boom()


def test_wrapped_exception_chains_the_original_as_cause() -> None:
    @wrap_backend_errors
    def boom() -> None:
        raise ValueError("original failure")

    with pytest.raises(StoreError) as exc_info:
        boom()
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == "original failure"


def test_an_existing_ragcore_error_passes_through_unwrapped() -> None:
    @wrap_backend_errors
    def boom() -> None:
        raise LoaderError("already typed, not a backend failure")

    with pytest.raises(LoaderError, match="already typed"):
        boom()


def test_store_error_itself_passes_through_unchanged() -> None:
    @wrap_backend_errors
    def boom() -> None:
        raise StoreError("intentional validation failure")

    with pytest.raises(StoreError, match="intentional validation failure") as exc_info:
        boom()
    assert exc_info.value.__cause__ is None  # not re-wrapped, so no new __cause__ added


def test_successful_call_returns_normally() -> None:
    @wrap_backend_errors
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5


def test_preserves_function_name_and_docstring() -> None:
    @wrap_backend_errors
    def documented() -> None:
        """A docstring."""

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "A docstring."


def test_error_message_includes_the_qualified_function_name() -> None:
    class Store:
        @wrap_backend_errors
        def method(self) -> None:
            raise RuntimeError("backend exploded")

    with pytest.raises(StoreError, match=r"Store\.method failed: backend exploded"):
        Store().method()


def test_base_exceptions_like_keyboard_interrupt_are_not_caught() -> None:
    @wrap_backend_errors
    def boom() -> None:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        boom()


def test_ragcore_error_is_still_an_exception_subtype_for_broad_catching() -> None:
    @wrap_backend_errors
    def boom() -> None:
        raise RuntimeError("x")

    with pytest.raises(RagcoreError):
        boom()
