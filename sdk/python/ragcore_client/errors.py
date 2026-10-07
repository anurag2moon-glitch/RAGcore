"""Exceptions raised by RagcoreClient."""

from __future__ import annotations


class RagcoreClientError(Exception):
    """Raised when the reference server returns a non-2xx response.
    `status_code` and `detail` (the server's own `{"detail": "..."}`
    error body, when present) are kept separately so a caller can branch
    on status_code without string-parsing the message."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"server returned {status_code}: {detail}")
